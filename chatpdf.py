from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain import PromptTemplate, LLMChain
from langchain.memory import ChatMessageHistory
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.llms import OpenAI


API_KEY = st.secrets["general"]["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"]


st.set_page_config(page_title='Mon prof particulier', layout="centered")

st.title('Mon prof particulier')
st.write('Importe ton cours en PDF !')

uploaded_file = st.file_uploader(
    'Importe ton cours en PDF !', type='pdf')

if uploaded_file is not None:
    # Write the uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_file_name = tmp.name

    # Load and split pages from PDF
    loader = PyPDFLoader(tmp_file_name)
    pages = loader.load_and_split()

    # The rest of your code

    # Don't forget to remove the temporary file
    os.remove(tmp_file_name)

    # Prepare chat model
    chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)

    # Prepare text splitter and split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50)
    documents = []
    for i in range(len(pages)):
        if len(pages[i].page_content) > 500:
            doc = text_splitter.split_documents([pages[i]])
            documents.extend(doc)
        else:
            documents.extend([pages[i]])

    # Prepare embeddings and embedding engine
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(documents, embeddings)
    retriever = db.as_retriever()

    # Prepare the prompt
    template = """
    Je suis étudiant. Tu dois m'aider à réviser mon cours en répondant à mes questions. Voici mon cours :
    COURS :
    {course}

    Très important : Si la réponse n'est pas dans le cours réponds "Je n'ai pas trouvé la réponse dans le cours"
    """
    prompt = PromptTemplate(input_variables=["course"], template=template)

    # Get user's question
    question = st.text_input("Pose moi tes questions !")

    if question:
        # Get relevant documents
        docs = retriever.get_relevant_documents(question)

        # Prepare relevant data for the prompt
        relevant_data = ""
        for d in docs:
            relevant_data += d.page_content+"\n\n "

        # Format the prompt
        setup = prompt.format(course=relevant_data)
        history = ChatMessageHistory()
        history.add_user_message(setup)
        history.add_ai_message("Compris. Pose tes questions !")

        # Get the answer from the model
        answer = chat([HumanMessage(content=question)]).content

        # Display the answer
        st.write(answer)
