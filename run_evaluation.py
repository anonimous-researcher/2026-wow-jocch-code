import argparse
import os
import json
import pandas
import logging
import utilities


def get_localized_template(template_entry, language):
    if not template_entry.get("localization", False):
        return template_entry["template"]
    return template_entry.get(language, template_entry["template"])


def dataset_kind(dataset_id: str) -> str:
    dataset_id = (dataset_id or "").lower()
    if "_c_" in dataset_id:
        return "concepts"
    if "_k_" in dataset_id:
        return "keywords"
    return "unknown"


def is_language_compatible(model_lang: str, dataset_lang: str) -> bool:
    model_lang = (model_lang or "").upper().strip()
    dataset_lang = (dataset_lang or "").upper().strip()
    if model_lang in ["MULTI", "ANY", "ALL", "X"]:
        return True
    return model_lang == dataset_lang


def decoder_defaults_l40(kind: str) -> dict:
    # L40 (48GB) defaults tuned for your text lengths
    if kind == "concepts":
        return {"batch_size": 32, "label_chunk_size": 19, "prompt_batch_size": 512, "max_length": 512}
    if kind == "keywords":
        return {"batch_size": 32, "label_chunk_size": 64, "prompt_batch_size": 512, "max_length": 512}
    return {"batch_size": 8, "label_chunk_size": 32, "prompt_batch_size": 256, "max_length": 512}


def pick(value, default):
    return default if value is None else value


def main():
    parser = argparse.ArgumentParser(description="Run evaluation (NLI templates; decoder-only single run).")

    parser.add_argument("--models_csv", required=True, type=str)
    parser.add_argument("--datasets_csv", required=True, type=str)
    parser.add_argument("--output_folder", required=True, type=str)
    parser.add_argument("--templates_json", required=True, type=str)
    parser.add_argument("--verbose", action="store_true")

    # Optional overrides (apply to decoder-only; NLI uses --batch_size_nli)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--label_chunk_size", type=int, default=None)
    parser.add_argument("--prompt_batch_size", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--decoder_label_template", type=str, default="{}")

    # NLI-only
    parser.add_argument("--batch_size_nli", type=int, default=8)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    models_df = pandas.read_csv(args.models_csv)
    datasets_df = pandas.read_csv(args.datasets_csv)
    with open(args.templates_json, "r", encoding="utf-8") as f:
        templates = json.load(f)

    os.makedirs(args.output_folder, exist_ok=True)

    if "TYPE" not in models_df.columns:
        models_df["TYPE"] = "NLI"
    models_df["TYPE"] = models_df["TYPE"].fillna("NLI").astype(str).str.upper()

    for _, drow in datasets_df.iterrows():
        data_id = str(drow["ID"])
        data_path = str(drow["PATH"])
        data_lang = str(drow["LANGUAGE"]).upper()
        df = pandas.read_csv(data_path)

        kind = dataset_kind(data_id)
        defaults = decoder_defaults_l40(kind)

        for _, mrow in models_df.iterrows():
            model_id = str(mrow["ID"])
            model_hf_id = str(mrow["MODEL"])
            model_type = str(mrow["TYPE"]).upper()
            model_lang = str(mrow["LANGUAGE"]).upper()

            if not is_language_compatible(model_lang, data_lang):
                if args.verbose:
                    logging.info(f"Skipping {model_id} ({model_lang}) on {data_id} ({data_lang})")
                continue

            if model_type == "DECODER":
                dec_batch = pick(args.batch_size, defaults["batch_size"])
                dec_chunk = pick(args.label_chunk_size, defaults["label_chunk_size"])
                dec_pbs = pick(args.prompt_batch_size, defaults["prompt_batch_size"])
                dec_maxlen = pick(args.max_length, defaults["max_length"])

                print(
                    f"\n🧪 {data_id} ({kind},{data_lang}) | {model_id} DECODER | "
                    f"bs={dec_batch} chunk={dec_chunk} pbs={dec_pbs} maxlen={dec_maxlen}"
                )

                results_df = utilities.decoder_only_keywords_extraction(
                    df=df,
                    model_id=model_hf_id,
                    verbose=args.verbose,
                    batch_size=dec_batch,
                    label_template=args.decoder_label_template,
                    label_chunk_size=dec_chunk,
                    max_length=dec_maxlen,
                    prompt_language=data_lang,
                    prompt_batch_size=dec_pbs
                )

                out_path = os.path.join(args.output_folder, f"{data_id}_{model_id}.csv")
                results_df.to_csv(out_path, index=False)
                print(f"✅ Saved: {out_path}")

            else:
                for t in templates:
                    tid = t.get("ID")
                    label_template = get_localized_template(t, data_lang)

                    print(f"\n🧪 {data_id} ({data_lang}) | {model_id} NLI | bs={args.batch_size_nli} | T={tid}")

                    results_df = utilities.keywords_extraction(
                        df=df,
                        model_id=model_hf_id,
                        verbose=args.verbose,
                        batch_size=args.batch_size_nli,
                        label_template=label_template
                    )

                    out_path = os.path.join(args.output_folder, f"{data_id}_{model_id}_{tid}.csv")
                    results_df.to_csv(out_path, index=False)
                    print(f"✅ Saved: {out_path}")


if __name__ == "__main__":
    main()
