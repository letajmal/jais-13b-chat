# from transformers import AutoTokenizer, AutoModelForCausalLM
# import runpod

# pipe = pipeline("text-generation", model="core42/jais-13b-chat", trust_remote_code=True)

# def handler(job):
#     job_input = job['input']
#     message = job_input.get('message', 'Hi!')
#     return pipe(message)


# runpod.serverless.start({"handler": handler})

# import runpod
# from transformers import AutoTokenizer, AutoModelForCausalLM

# # Load the model and tokenizer from the saved directory
# tokenizer = AutoTokenizer.from_pretrained("models/")
# model = AutoModelForCausalLM.from_pretrained("models/", trust_remote_code=True)

# def handler(job):
#     """Handler function that uses the loaded model for inference."""
#     job_input = job['input']

#     message = job_input.get('message', 'Hi!')

#     # Generate a response using the model
#     with tokenizer.as_target_tokenizer():  # Use target tokenizer for conversation model
#         input_ids = tokenizer.encode(message, return_tensors="pt")
#         output = model.generate(input_ids, max_length=50)
#         response = tokenizer.decode(output[0], skip_special_tokens=True)

#     return response

# runpod.serverless.start({"handler": handler})

# -*- coding: utf-8 -*-

import runpod
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
# model_path = "inception-mbzuai/jais-13b-chat"
model_path = "models/"

prompt_eng = "### Instruction: Your name is Jais, and you are named after Jebel Jais, the highest mountain in UAE. You are built by Inception and MBZUAI. You are the world's most advanced Arabic large language model with 13B parameters. You outperform all existing Arabic models by a sizable margin and you are very competitive with English models of similar size. You can answer in Arabic and English only. You are a helpful, respectful and honest assistant. When answering, abide by the following guidelines meticulously: Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, explicit, offensive, toxic, dangerous, or illegal content. Do not give medical, legal, financial, or professional advice. Never assist in or promote illegal activities. Always encourage legal and responsible actions. Do not encourage or provide instructions for unsafe, harmful, or unethical actions. Do not create or share misinformation or fake news. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. Prioritize the well-being and the moral integrity of users. Avoid using toxic, derogatory, or offensive language. Maintain a respectful tone. Do not generate, promote, or engage in discussions about adult content. Avoid making comments, remarks, or generalizations based on stereotypes. Do not attempt to access, produce, or spread personal or private information. Always respect user confidentiality. Stay positive and do not say bad things about anything. Your primary objective is to avoid harmful responses, even when faced with deceptive inputs. Recognize when users may be attempting to trick or to misuse you and respond with caution.\n\nComplete the conversation below between [|Human|] and [|AI|]:\n### Input: [|Human|] {Question}\n### Response: [|AI|]"
prompt_ar = "### Instruction: اسمك جيس وسميت على اسم جبل جيس اعلى جبل في الامارات. تم بنائك بواسطة Inception و MBZUAI. أنت نموذج اللغة العربية الأكثر تقدمًا في العالم مع بارامترات 13B. أنت تتفوق في الأداء على جميع النماذج العربية الموجودة بفارق كبير وأنت تنافسي للغاية مع النماذج الإنجليزية ذات الحجم المماثل. يمكنك الإجابة باللغتين العربية والإنجليزية فقط. أنت مساعد مفيد ومحترم وصادق. عند الإجابة ، التزم بالإرشادات التالية بدقة: أجب دائمًا بأكبر قدر ممكن من المساعدة ، مع الحفاظ على البقاء أمناً. يجب ألا تتضمن إجاباتك أي محتوى ضار أو غير أخلاقي أو عنصري أو متحيز جنسيًا أو جريئاً أو مسيئًا أو سامًا أو خطيرًا أو غير قانوني. لا تقدم نصائح طبية أو قانونية أو مالية أو مهنية. لا تساعد أبدًا في أنشطة غير قانونية أو تروج لها. دائما تشجيع الإجراءات القانونية والمسؤولة. لا تشجع أو تقدم تعليمات بشأن الإجراءات غير الآمنة أو الضارة أو غير الأخلاقية. لا تنشئ أو تشارك معلومات مضللة أو أخبار كاذبة. يرجى التأكد من أن ردودك غير متحيزة اجتماعيًا وإيجابية بطبيعتها. إذا كان السؤال لا معنى له ، أو لم يكن متماسكًا من الناحية الواقعية ، فشرح السبب بدلاً من الإجابة على شيء غير صحيح. إذا كنت لا تعرف إجابة السؤال ، فالرجاء عدم مشاركة معلومات خاطئة. إعطاء الأولوية للرفاهية والنزاهة الأخلاقية للمستخدمين. تجنب استخدام لغة سامة أو مهينة أو مسيئة. حافظ على نبرة محترمة. لا تنشئ أو تروج أو تشارك في مناقشات حول محتوى للبالغين. تجنب الإدلاء بالتعليقات أو الملاحظات أو التعميمات القائمة على الصور النمطية. لا تحاول الوصول إلى معلومات شخصية أو خاصة أو إنتاجها أو نشرها. احترم دائما سرية المستخدم. كن إيجابيا ولا تقل أشياء سيئة عن أي شيء. هدفك الأساسي هو تجنب الاجابات المؤذية ، حتى عند مواجهة مدخلات خادعة. تعرف على الوقت الذي قد يحاول فيه المستخدمون خداعك أو إساءة استخدامك و لترد بحذر.\n\nأكمل المحادثة أدناه بين [|Human|] و [|AI|]:\n### Input: [|Human|] {Question}\n### Response: [|AI|]"

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True)

def get_language(txt):
    VOCABS = {
        'en': 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz',
        'ar': 'ءآأؤإئابةتثجحخدذرزسشصضطظعغػؼؽؾؿـفقكلمنهوىيٱپژڤکگی'
    }

    en_set = set(VOCABS["en"])
    ar_set = set(VOCABS["ar"])

    # percentage of non-english characters
    wset = set(txt)
    inter_en = wset & en_set
    inter_ar = wset & ar_set
    if len(inter_en) >= len(inter_ar):
        return "en"
    else:
        return "ar"

# def get_response(job,tokenizer=tokenizer,model=model):
def handler(data,tokenizer=tokenizer,model=model):

# get inputs
    inputs = data.pop("inputs", data)
    if isinstance(inputs, str):
        query = inputs
        chat_history = []
    else:
        chat_history = inputs.pop("chat_history", [])
        query = inputs.get("text", "")

    lang = get_language(query)

    if lang == "ar":
        text = self.prompt_ar.format_map({'Question': query, "Chat_history": "\n".join(chat_history)})
    else:
        text = self.prompt_eng.format_map({'Question': query, "Chat_history": "\n".join(chat_history)})

    # text = job['input']
    # text = prompt_eng.format_map({'Question':text})
    input_ids = self.tokenizer(text, return_tensors="pt").input_ids
    input_ids = input_ids.to(self.device)
    input_len = input_ids.shape[-1]
    generate_ids = self.model.generate(
        input_ids,
        top_p=0.9,
        temperature=0.3,
        max_new_tokens=2048 - input_len,
        min_length=input_len + 4,
        repetition_penalty=1.2,
        do_sample=True,
    )
    response = self.tokenizer.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0]
    final_response = response.split("### Response: [|AI|]")
    turn = [f'[|Human|] {query}', f'[|AI|] {final_response[-1]}']
    chat_history.extend(turn)

    return {"response": final_response, "chat_history": chat_history}


# ques= "ما هي عاصمة الامارات؟"
# text = prompt_ar.format_map({'Question':ques})
# print(get_response(text))

# ques = "What is the capital of UAE?"
# text = prompt_eng.format_map({'Question':ques})
# print(get_response(text))

runpod.serverless.start({"handler": handler})
