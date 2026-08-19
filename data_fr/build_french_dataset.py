#!/usr/bin/env python3
"""
build_french_dataset_final.py
Dataset français : Wikipedia + Common Corpus + Books.
AUCUN nettoyage manuel. Déduplication MD5 globale.
"""

import os
import gzip
import json
import hashlib
import argparse
from collections import Counter

from datasets import load_dataset
from tqdm import tqdm
from huggingface_hub import login
login(token=os.environ.get("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"))



def md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def is_valid(text: str) -> bool:
    return text and 200 <= len(text) <= 50000


def is_french(doc) -> bool:
    lang = doc.get("language") or doc.get("lang", "")
    if isinstance(lang, str):
        return lang.lower() in ("french", "fr", "fra")
    elif isinstance(lang, list):
        return any(str(l).lower() in ("french", "fr", "fra") for l in lang)
    return False


def get_text(doc, source: str) -> str:
    if source == "books":
        return doc.get("complete_text", "") or doc.get("text", "")
    return doc.get("text", "")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/frenchmix_raw")
    parser.add_argument("--common_limit", type=int, default=7_000_000)
    parser.add_argument("--test", action="store_true",
                        help="Test rapide : 100k common max pour vérifier le token")
    args = parser.parse_args()

    if args.test:
        args.common_limit = 100_000
        print("🧪 MODE TEST : common_limit=100k")

    os.makedirs(args.output, exist_ok=True)
    out_path = os.path.join(args.output, "frenchmix_clean.jsonl.gz")
    seen_hashes = set()
    stats = Counter()
    written = 0

    with gzip.open(out_path, "wt", encoding="utf-8") as fout:

        # 1. WIKIPEDIA
        print("[1/3] Wikipedia fr...")
        for config in ["20231101.fr", "20220301.fr", "20210401.fr"]:
            try:
                ds = load_dataset("wikimedia/wikipedia", config, split="train", streaming=True)
                break
            except Exception as e:
                print(f"  {config}: {e}")
                continue
        else:
            raise RuntimeError("Aucune config Wikipedia disponible")

        for doc in tqdm(ds, desc="wiki"):
            text = get_text(doc, "wiki")
            if not is_valid(text):
                stats["wiki_filtered"] += 1
                continue
            h = md5(text)
            if h in seen_hashes:
                stats["wiki_dup"] += 1
                continue
            seen_hashes.add(h)
            fout.write(json.dumps({"text": text, "source": "wiki"}, ensure_ascii=False) + "\n")
            written += 1
            stats["wiki"] += 1
        print(f"  → {stats['wiki']:,} docs wiki")

        # 2. COMMON CORPUS
        print(f"\n[2/3] Common Corpus fr (max {args.common_limit:,})...")
        ds = load_dataset("PleIAs/common_corpus", split="train", streaming=True)
        common_cnt = 0

        for doc in tqdm(ds, desc="common"):
            if common_cnt >= args.common_limit:
                break
            if not is_french(doc):
                stats["common_not_fr"] += 1
                continue
            text = get_text(doc, "common")
            if not is_valid(text):
                stats["common_filtered"] += 1
                continue
            h = md5(text)
            if h in seen_hashes:
                stats["common_dup"] += 1
                continue
            seen_hashes.add(h)
            fout.write(json.dumps({"text": text, "source": "common"}, ensure_ascii=False) + "\n")
            written += 1
            common_cnt += 1
            stats["common"] += 1
        print(f"  → {stats['common']:,} docs common")

        # 3. BOOKS
        print("\n[3/3] French-PD-Books...")
        ds = load_dataset("PleIAs/French-PD-Books", split="train", streaming=True)

        for doc in tqdm(ds, desc="books"):
            text = get_text(doc, "books")
            if not is_valid(text):
                stats["books_filtered"] += 1
                continue
            h = md5(text)
            if h in seen_hashes:
                stats["books_dup"] += 1
                continue
            seen_hashes.add(h)
            fout.write(json.dumps({"text": text, "source": "books"}, ensure_ascii=False) + "\n")
            written += 1
            stats["books"] += 1
        print(f"  → {stats['books']:,} docs books")

    # RÉSUMÉ
    print(f"\n{'='*60}")
    print(f"✅ Fichier : {out_path}")
    print(f"   Total docs uniques : {written:,}")
    for src in ["wiki", "common", "books"]:
        print(f"   {src:10s} : {stats[src]:,} retenus")
    not_fr = stats.get("common_not_fr", 0)
    if not_fr:
        print(f"   common_not_fr : {not_fr:,} non-français rejetés")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()