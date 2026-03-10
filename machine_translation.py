import csv

import pandas
import googletrans
import asyncio
import easynmt


async def translate_bulk(sentences: list):
    async with googletrans.Translator() as translator:
        translations = await translator.translate(sentences, dest='en', src="it")
        return [translation.text for translation in translations]


trans_key_df = pandas.read_csv("data/k_translation.csv")
trans_con_df = pandas.read_csv("data/c_translation.csv")

it_keywords_list = trans_key_df["IT"].to_list()
it_concepts_list = trans_con_df["IT"].to_list()
en_keywords_list = trans_key_df["EN"].to_list()
en_concepts_list = trans_con_df["EN"].to_list()

k_df_it = pandas.read_csv("data/quotes_k_it.csv")
c_df_it = pandas.read_csv("data/quotes_c_it.csv")

text_list = k_df_it["text"].to_list()

en_texts = asyncio.run(translate_bulk(text_list))

model = easynmt.EasyNMT('opus-mt')
en_nmt_text = model.translate(text_list, target_lang='en', source_lang="it")

key_rename_map = {it_keywords_list[i]: en_keywords_list[i] for i in range(len(it_keywords_list))}
con_rename_map = {it_concepts_list[i]: en_concepts_list[i] for i in range(len(it_concepts_list))}

k_df_en = k_df_it.rename(key_rename_map, axis=1)
k_df_en["text"] = en_texts
c_df_en = c_df_it.rename(con_rename_map, axis=1)
c_df_en["text"] = en_texts

k_df_en_nmt = k_df_it.rename(key_rename_map, axis=1)
k_df_en_nmt["text"] = en_nmt_text
c_df_en_nmt = c_df_it.rename(con_rename_map, axis=1)
c_df_en_nmt["text"] = en_nmt_text

k_df_en.to_csv("data/quotes_k_en.csv", index=False, quoting=csv.QUOTE_NONNUMERIC)
c_df_en.to_csv("data/quotes_c_en.csv", index=False, quoting=csv.QUOTE_NONNUMERIC)

k_df_en_nmt.to_csv("data/quotes_k_en_nmt.csv", index=False, quoting=csv.QUOTE_NONNUMERIC)
c_df_en_nmt.to_csv("data/quotes_c_en_nmt.csv", index=False, quoting=csv.QUOTE_NONNUMERIC)
