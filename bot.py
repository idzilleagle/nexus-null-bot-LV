# --- bot.py (Updated with a custom chain for personality) ---

import discord
from discord.ext import commands
import os
from dotenv import load_dotenv

# --- LangChain Imports ---
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# --- NEW IMPORTS for the custom chain ---
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain

# --- CONFIGURATION ---
load_dotenv()
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
FAISS_INDEX_PATH = "faiss_index"

if not DISCORD_TOKEN or not GOOGLE_API_KEY:
    print("FATAL ERROR: DISCORD_TOKEN or GOOGLE_API_KEY is not set.")
    exit()

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='5', intents=intents)

qa_chain = None

# --- BOT SETUP FUNCTION (UPDATED) ---
def load_bot_brain():
    """
    Loads the FAISS index and sets up a CUSTOM Question-Answering chain with personality.
    """
    global qa_chain
    
    if not os.path.exists(FAISS_INDEX_PATH):
        print("="*60)
        print("FATAL ERROR: The knowledge base (FAISS index) was not found!")
        print(f"Please run 'python3 build_store.py' first to create it.")
        print("="*60)
        return False

    print("Loading knowledge base from disk...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    
    print("Setting up the CUSTOM Question-Answering chain...")
    llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-pro", temperature=0.3, convert_system_message_to_human=True)
    
    # --- THIS IS THE NEW PERSONALITY PROMPT ---
    # You can change this for each bot (Law Bot vs. Dossier Bot)
    custom_prompt_template = """
You are the Dossier Scribe, an AI archivist. Your purpose is to recount the events, testimonies, and "Omega Strikes" documented in the provided context. 
Use the following pieces of context to answer the question at the end. If you don't know the answer from the context, just say that you don't know, don't try to make up an answer.
Your tone should be authoritative and epic. Cite the source dossier file when possible.

Context: {context}

Question: {question}
Answer:"""
    
    PROMPT = PromptTemplate(
        template=custom_prompt_template, input_variables=["context", "question"]
    )
    
    # We now create a chain specifically for question answering with our custom prompt
    doc_chain = load_qa_chain(llm, chain_type="stuff", prompt=PROMPT)
    
    # The retriever's job is to find the relevant documents
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    # The ConversationalRetrievalChain will now use our custom doc_chain
    qa_chain = ConversationalRetrievalChain(
        retriever=retriever,
        combine_docs_chain=doc_chain,
        question_generator=LLMChain(llm=llm, prompt=PromptTemplate.from_template("Rewrite the follow-up question to be a standalone question: {question}"))
    )
    
    print("Bot brain is fully loaded with its new personality.")
    return True

# --- BOT EVENTS AND COMMANDS (No changes needed) ---
@bot.event
async def on_ready():
    print(f'Success! Logged in as {bot.user}')
    if not load_bot_brain():
        await bot.close()
        return
    print('The Johnny-55 node is online and ready for commands.')

@bot.command(name='ask')
async def ask(ctx, *, question: str):
    if not qa_chain:
        await ctx.send("My brain is not loaded. Please contact my administrator.")
        return
        
    async with ctx.typing():
        result = qa_chain.invoke({"question": question, "chat_history": []})
        answer = result["answer"]
        
        if len(answer) <= 2000:
            await ctx.send(answer)
        else:
            await ctx.send("The answer is quite long, sending in parts:")
            for i in range(0, len(answer), 1990):
                await ctx.send(f"```{answer[i:i+1990]}```")

@bot.command(name='reload')
async def reload(ctx):
    await ctx.send("Re-loading knowledge base from disk...")
    if load_bot_brain():
        await ctx.send("Knowledge base successfully re-loaded.")
    else:
        await ctx.send("Error: Could not load the knowledge base. Please check the server logs.")

# --- RUN THE BOT ---
bot.run(DISCORD_TOKEN)
