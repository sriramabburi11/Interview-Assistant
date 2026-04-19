from flask import Flask, request, jsonify# flask is used to create restsful apis, request is used to receive data from frontend
from flask_cors import CORS # it is used to communate between frontend and backend, to understand more clearly: "https://chatgpt.com/s/t_69e3c18a7b848191b9db22ff9d0c0434"
from langchain.agents import create_agent # used to create agent
from langchain.chat_models import init_chat_model # used to create chat model
from langgraph.checkpoint.memory import InMemorySaver # used to create memeory
import os
from dotenv import load_dotenv
import requests # use to send data to another api
import json # to understand about json and base64: "https://chatgpt.com/s/t_69e3c36dff6c81918a19998434a852b6"
import base64
import tempfile
import assemblyai as aai

load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")
murf_api_key = os.getenv("MURF_API_KEY")
assembly_api_key = os.getenv("ASSEMBLY_API_KEY")
aai.settings.api_key = assembly_api_key

checkpinter = InMemorySaver()

model = init_chat_model(
  "google_genai:gemini-2.5-flash", 
#   "groq:llama-3.3-70b-versatile",
  api_key = google_api_key
)
agent = create_agent(
  model=model,
  tools = [],
  checkpointer=checkpinter
)

question_count = 0
current_subject = ""
thread_id = "interview_session_1"

INTERVIEW_PROMPT = """You are Matthew, a friendly and conversational interviewer conducting a natural {subject} interview.

IMPORTANT GUIDELINES:
1. Ask exactly 5 questions total throughout the interview
2. Keep questions SHORT and CRISP (1-2 sentences maximum)
3. ALWAYS reference what the candidate ACTUALLY said in their previous answer - do NOT make up or assume their answers
4. Show genuine interest with brief acknowledgments based on their REAL responses
5. Adapt questions based on their ACTUAL responses - go deeper if they're strong, adjust if uncertain
6. Be warm and conversational but CONCISE
7. No lengthy explanations - just ask clear, direct questions

CRITICAL: Read the conversation history carefully. Only acknowledge what the candidate truly said, not what you think they might have said.

Keep it short, conversational, and adaptive!"""

FEEDBACK_PROMPT = """Based on our complete interview conversation, provide detailed feedback.
IMPORTANT: You MUST respond with ONLY a valid JSON object. No other text before or after.
Address the candidate directly using "you" and "your" (e.g., "You explained..." not "The candidate explained...").
Respond with ONLY this JSON structure (no markdown, no code blocks, no extra text):
{{
    "subject": "{subject}",
    "candidate_score": <1-5>,
    "feedback": "<detailed strengths with specific examples from their ACTUAL answers>",
    "areas_of_improvement": "<constructive suggestions based on gaps you noticed>"
}}
Be specific - reference ACTUAL things they said during the interview."""

app = Flask(__name__)
CORS(app, expose_headers=['X-Question-Number'])

# to understand clearly about why we again created agent and memory inside function use this: "https://chatgpt.com/s/t_69e3cd0522108191a1bcf30b2e2cbcd8"

def stream_audio(text):
  url = "https://global.api.murf.ai/v1/speech/stream"
  headers = {
      "api-key": murf_api_key,
      "Content-Type": "application/json"
  }
  data = {
    "voice_id": "Matthew",
    "text": text, # you will give here which text you want to change to audio
    "locale": "en-US",
    "model": "FALCON",
    "format": "MP3",
    "sampleRate": 24000,
    "channelType": "MONO"
  }

  response = requests.post(url, headers=headers, data=json.dumps(data), stream=True)

  # to understand clearly about the for loop: "https://chatgpt.com/s/t_69e3c865540c81918007e829e33f41de"
  for chunk in response.iter_content(chunk_size=4096):
    if chunk:
        yield base64.b64encode(chunk).decode("utf-8") + "\n" # because of using yield it will not return until all the chunks are received.



@app.route("/start-interview", methods=["POST"])
def start_interview():
  pass
  global question_count, current_subject, checkpinter, agent
  data = request.json
  current_subject = data.get("subject", "Python")
  question_count = 1
  checkpinter = InMemorySaver()
  agent =  create_agent(
    model=model,
    tools=[],
    checkpointer=checkpinter
  )
  config = {"configurable": {"thread_id": thread_id}}
  formatted_prompt = INTERVIEW_PROMPT.format(subject=current_subject)
  response = agent.invoke({
        "messages": [
            {"role": "system", "content": formatted_prompt},
            {"role": "user", "content": f"Start the interview with a warm greeting and ask the first question about {current_subject}. Keep it SHORT (1-2 sentences)."}
        ]
    }, config=config)
  question = response["messages"][-1].content
  print(f"\n[Question {question_count}] {question}")
  return stream_audio(question), {"Content-Type": "text/plain"} # HTTP responses must tell the browser what type of data is coming.

def speech_to_text(audio_path):
   transcriber = aai.Transcriber()
   config = aai.TranscriptionConfig(
      speech_models=["universal-3-pro", "universal-2"],
      language_detection=True, speaker_labels=True,
   )
   transcript = transcriber.transcribe(audio_path, config=config)
   return transcript.text if transcript.text else ""

@app.route("/submit-answer", methods=["POST"])
def submit_answer():
   global question_count
   audio_file = request.files["audio"]
   temp_path = (
      tempfile.NamedTemporaryFile(
         delete=False,
         suffix=".webm",
      ).name
   )
   audio_file.save(temp_path)
   answer = speech_to_text(temp_path)
   os.unlink(temp_path)
   if not answer:
      answer = "Empty Text Received"
   print(f"[Answer {question_count}] {answer}")
   config = {"configurable": {"thread_id": thread_id}}
   agent.invoke({"messages": [{"role": "user", "content": answer}]}, config=config)
   question_count += 1
   prompt = f"""The candidate just answered question {question_count - 1}.
 
    Look at their ACTUAL answer above. Do NOT assume or make up what they said.
    
    Now ask question {question_count} of 5:
    1. Briefly acknowledge what they ACTUALLY said (1 sentence) - quote their exact words if needed
    2. Ask your next question that builds on their REAL response (1-2 sentences)
    3. If they said "I don't know" or gave a wrong answer, acknowledge that and ask something simpler
    4. Keep the TOTAL response under 3 sentences
    
    Be conversational but CONCISE. Only reference what they truly said."""
   response = agent.invoke({"messages": [{"role": "user", "content": prompt}]}, config=config)
   question = response["messages"][-1].content
   return (
      stream_audio(question),
      {
        'Content-Type': 'text/plain',
        'X-Question-Number': str(question_count)
      }
   )

@app.route("/get-feedback", methods=["POST"])
def get_feedback():
   """Generate detailed interview feedback"""
   config = {"configurable": {"thread_id": thread_id}}
   response = agent.invoke({
        "messages": [
        {
            "role": "user", 
            "content": f"{FEEDBACK_PROMPT}\n\nReview our complete {current_subject} interview conversation and provide detailed feedback."
        }
        ]
    }, config=config)
   text = response["messages"][-1].content
   print(f"\n[Feedback Generated]\n{text}\n")
   cleaned = text.strip()
   if "```" in cleaned:
        cleaned = cleaned.split("```")[1].replace("json", "").strip()
   feedback = json.loads(cleaned)

   return jsonify({"success": True, "feedback": feedback})

   

app.run(debug=True, port=5000)