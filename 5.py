import base64
import os
from google import genai
from google.genai import types


def generate():
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    model = "gemini-2.0-flash"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text="""هلا"""),
            ],
        ),
        types.Content(
            role="model",
            parts=[
                types.Part.from_text(text="""أهلاً بك! كيف يمكنني مساعدتك اليوم؟
"""),
            ],
        ),
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text="""تدريبي على القرائه
"""),
            ],
        ),
        types.Content(
            role="model",
            parts=[
                types.Part.from_text(text="""حسناً، يسعدني مساعدتك في التدرب على القراءة. لنجعل الأمر ممتعاً وفعالاً، إليك بعض الخيارات التي يمكننا تجربتها:

**1. قراءة نصوص قصيرة ومناقشتها:**

*   يمكنني أن أقدم لك نصوصاً قصيرة في مواضيع مختلفة، مثل الأخبار، القصص القصيرة، أو المقالات البسيطة.
*   بعد قراءة النص، يمكننا مناقشة معانيه، طرح أسئلة حوله، أو تلخيصه.
*   هذا يساعدك على فهم المقروء وتحسين مهاراتك في التحليل.

**2. قراءة كلمات وجمل بصوت عالٍ:**

*   يمكنني أن أعرض لك كلمات وجمل، وتتدرب على قراءتها بصوت عالٍ.
*   يمكنني تصحيح نطقك إذا لزم الأمر، وتقديم ملاحظات لتحسينه.
*   هذا يساعدك على تحسين طلاقتك في القراءة ونطق الكلمات بشكل صحيح.

**3. قراءة نصوص موجهة للمبتدئين:**

*   هناك العديد من النصوص الموجهة خصيصاً للمبتدئين في القراءة، والتي تستخدم كلمات بسيطة وتراكيب سهلة.
*   يمكنني أن أقدم لك بعض هذه النصوص، ونقرأها معاً خطوة بخطوة.
*   هذا يساعدك على بناء الثقة بالنفس وتحقيق التقدم بشكل تدريجي.

**4. استخدام الألعاب والتطبيقات التعليمية:**

*   هناك العديد من الألعاب والتطبيقات التعليمية التي تجعل تعلم القراءة ممتعاً وتفاعلياً.
*   يمكنني أن أرشح لك بعض هذه الأدوات، واستخدامها كجزء من تدريبنا.

**للبدء، ما هو مستواك في القراءة؟ وهل لديك تفضيلات معينة لنوع النصوص التي تود قراءتها؟**

على سبيل المثال، هل تفضل:

*   **نصوص بسيطة جداً للمبتدئين؟**
*   **قصص قصيرة للأطفال؟**
*   **مقالات بسيطة عن مواضيع معينة تهمك؟**

كلما عرفت المزيد عن تفضيلاتك ومستواك، كلما تمكنت من تقديم المساعدة الأفضل."""),
            ],
        ),
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text="""INSERT_INPUT_HERE"""),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        top_p=0.95,
        top_k=40,
        max_output_tokens=8192,
        safety_settings=[
            types.SafetySetting(
                category="HARM_CATEGORY_HARASSMENT",
                threshold="BLOCK_NONE",  # Block none
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_HATE_SPEECH",
                threshold="BLOCK_NONE",  # Block none
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                threshold="BLOCK_NONE",  # Block none
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="BLOCK_NONE",  # Block none
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_CIVIC_INTEGRITY",
                threshold="BLOCK_NONE",  # Block none
            ),
        ],
        response_mime_type="text/plain",
    )

    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        print(chunk.text, end="")

if __name__ == "__main__":
    generate()
