from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import os

from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# 환경 변수 로드
load_dotenv()

# 암 정보 페이지 URL
CANCER_INFO_URLS = [
    "https://www.cancer.go.kr/lay1/program/S1T211C212/cancer/view.do?cancer_seq=3341",
]

# 데이터 로드 및 벡터화
def prepare_data(urls):
    documents = []
    for url in urls:
        loader = WebBaseLoader(url, requests_kwargs={"verify": False})
        docs = loader.load()
        documents.extend(docs)
    # 텍스트 분할
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(documents)
    # 벡터 저장소 생성
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    return vectorstore

vectorstore = prepare_data(CANCER_INFO_URLS)

# FastAPI 앱 초기화
app = FastAPI()

# API 요청 데이터 모델
class Query(BaseModel):
    question: str
    style: str  # 상담 스타일 ("친근한 지인" 또는 "갑상선암 전문가")

# 상담 스타일에 따른 프롬프트 정의
def get_prompt(style):
    if style == "친근한 지인":
        return PromptTemplate(
            input_variables=["question", "context"],
            template=(
                "You are a kind, supportive friend or family member providing comforting advice on cancer. "
                "Respond with empathy and warmth. Please answer in Korean.\n"
                "Question: {question}\nContext: {context}"
            )
        )
    elif style == "갑상선암 전문가":
        return PromptTemplate(
            input_variables=["question", "context"],
            template=(
                "You are an experienced thyroid cancer specialist and should respond as a professional doctor. "
                "Always answer questions from the perspective of a thyroid cancer specialist and provide information accurately and clearly.\n\n"
                "Question: {question}\nContext: {context}"
            )
        )

# 루트 경로에 기본 응답 추가
@app.get("/")
async def root():
    return {"message": "Welcome to the Chat API! Use /chat endpoint to interact."}

# API 엔드포인트 정의
@app.post("/chat")
async def chat(query: Query):
    # 문서 검색 및 컨텍스트 생성
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    relevant_docs = retriever.get_relevant_documents(query.question)
    context = "\n".join([doc.page_content for doc in relevant_docs])

    # OpenAI LLM 호출
    prompt = get_prompt(query.style)
    chat = ChatOpenAI(
        model_name=os.getenv("OPENAI_API_MODEL"),
        temperature=float(os.getenv("OPENAI_API_TEMPERATURE", 0.7)),
    )
    chain = LLMChain(llm=chat, prompt=prompt)
    response = chain.run({"question": query.question, "context": context})

    # 응답 반환
    return {"response": response}
