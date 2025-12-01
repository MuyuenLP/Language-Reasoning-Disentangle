import re
import os
from tqdm import tqdm
import fasttext



class fidelity_evaluator:
    def __init__(self):
        self.model = fasttext.load_model("/home/jhguo/Code/FreeEvalLM/glotlid/model.bin")
        # self.init_languages()
        self.init_mapping()

    def init_languages(self):
        self.languages_inputs = {
            "en": "This is a test sentence in English.",
            "zh": "这是中文测试句子。",
            "ja": "これは日本語のテスト文です。",
            "fr": "Ceci est une phrase de test en français.",
            "de": "Dies ist ein Testsatz auf Deutsch.",
            "es": "Esta es una frase de prueba en español.",
            "bn": "এটি বাংলায় একটি পরীক্ষামূলক বাক্য।",
            "sw": "Hii ni sentensi ya majaribio kwa Kiswahili.",
            "ar": "هذه جملة اختبار باللغة العربية.",
            "ru": "Это тестовое предложение на русском языке.",
            "te": "ఇది తెలుగు లో ఒక పరీక్షా వాక్యం.",
            "th": "นี่คือประโยคทดสอบในภาษาไทย。",
        }
        
    def init_mapping(self):
        self.language_mapping = {
            "en": "__label__eng_Latn",
            "zh": "__label__cmn_Hani",
            "ja": "__label__jpn_Jpan",
            "es": "__label__spa_Latn",
            "fr": "__label__fra_Latn",
            "de": "__label__deu_Latn",
            "sw": "__label__swh_Latn",
            "bn": "__label__ben_Beng",
            "ru": "__label__rus_Cyrl",
            "te": "__label__tel_Telu",
            "th": "__label__tha_Thai",
        }

    def evaluate(self, questions, answers, name):
        results = []
        for i, (question, answer) in enumerate(tqdm(zip(questions, answers), total=len(questions))):
            question = question.replace("\n", " ")
            answer = answer.replace("\n", " ")
            # question_lang = self.model.predict(question)[0][0].split("_")[-1]
            # question_lang = self.model.predict(self.languages_inputs[name])[0][0]
            # print(question_lang)
            question_lang = self.language_mapping[name]
            answer_lang = self.model.predict(answer)[0][0]
            # print(f"Response {i+1}/{len(questions)}: {answer_lang} vs {question_lang}")
            results.append(answer_lang == question_lang)
        # return {"Language fidelity": sum(results) / len(results)}
        return results
    
    

if __name__ == "__main__":
    a = fidelity_evaluator()
    a.evaluate(["hello world", "你好，世界"], ["hello world", "hello world"], "th")