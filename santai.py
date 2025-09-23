# app.py
import streamlit as st
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.schema import Document
import Levenshtein as fuzz
import os
from dotenv import load_dotenv
import re

# Load API Key
load_dotenv()
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
except:
    groq_api_key = os.getenv('GROQ_API_KEY')

# Function to load and process our knowledge base
@st.cache_resource
def build_knowledge_base():
    documents = []
    
    # 1. Load data from the comprehensive regulations file
    try:
        with open('data/peraturan.txt', 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Split the content into meaningful sections
            sections = content.split('BAB ')[1:]  # Skip the header part
            
            for i, section in enumerate(sections):
                # Add each major section as a document
                if section.strip():
                    bab_title = section.split('\n')[0].strip()
                    documents.append(Document(
                        page_content=f"BAB {i+1}: {bab_title}\n\n{section[:2000]}"  # Limit content length
                    ))
            
            # Also add the complete file as a reference document
            documents.append(Document(
                page_content=f"PANDUAN LENGKAP VERIFIKASI PINJOL - Sumber: peraturan.txt\n\n{content[:4000]}"
            ))
            
    except FileNotFoundError:
        st.sidebar.error("peraturan.txt not found. Please create this file in data/ directory.")
        return None

    # 2. Check if we have any documents
    if not documents:
        st.sidebar.error("No documents were loaded. Please check your data files.")
        return None

    # 3. Create Embeddings with better error handling
    try:
        # Use a more reliable embedding model
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},  # Force CPU to avoid GPU issues
            encode_kwargs={'normalize_embeddings': False}
        )
        
        # Test the embedding model with a simple text
        test_embedding = embeddings.embed_query("test")
        if len(test_embedding) == 0:
            st.sidebar.error("Embedding model failed to generate vectors.")
            return None
            
    except Exception as e:
        st.sidebar.error(f"Error loading embedding model: {e}")
        return None

    # 4. Create Vector Store
    try:
        vectorstore = FAISS.from_documents(documents, embeddings)
        st.sidebar.success("Knowledge base loaded successfully from peraturan.txt!")
        return vectorstore
    except Exception as e:
        st.sidebar.error(f"Error creating vector store: {e}")
        return None


# Build the knowledge base
vectorstore = build_knowledge_base()


# Initialize the Groq LLM (Let's use the fast Llama 3 8B for prototyping)
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")


# Initialize session state for user data and conversation flow
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.user_data = {
        'name': None, 'age': None, 'job': None, 'income': None,
        'lenders': [],  # List to store multiple lenders
        'data_complete': False
    }
    st.session_state.current_step = 'name'
    st.session_state.analysis_shown = False
    st.session_state.should_generate_analysis = False
    st.session_state.analyzing_msg = None
    st.session_state.current_lender = {}  # Temporary storage for current lender
    st.session_state.add_another_lender = None  # Track if user wants to add more
    st.session_state.confirmation_step = False  # Track if we're in confirmation step
    st.session_state.correcting_lender = False  # Track if user is correcting lender names
    st.session_state.lender_to_correct = None  # Track which lender is being corrected


# Set page config
st.set_page_config(
    page_title="SantAI - AI yang bikin tenang, aman, dan nggak panik ketika terlilit utang!", 
    page_icon="logo.png",
    layout="wide")

# Create columns for logo and title
col1, col2 = st.columns([1, 4])
with col1:
    st.image("logo.png", width=256)  # Adjust width as needed
with col2:
    st.header("SantAI - Teman anda untuk bikin tenang, aman, dan nggak panik ketika terlilit utang!")
    st.caption("Powered by LLaMA via Groq & OJK Data")

# Alternatively, in sidebar
with st.sidebar:
    st.image("logo.png", width=100)
    st.text("AI yang bikin tenang, aman, dan nggak panik ketika terlilit utang!")


def get_next_question():
    """Determine which question to ask next based on current step"""
    current_step = st.session_state.current_step
    
    # Define the flow for the first lender
    if current_step == 'name':
        return 'age'
    elif current_step == 'age':
        return 'job'
    elif current_step == 'job':
        return 'income'
    elif current_step == 'income':
        return 'lender_name'
    elif current_step == 'lender_name':
        return 'lender_website'
    elif current_step == 'lender_website':
        return 'loan_amount'
    elif current_step == 'loan_amount':
        return 'monthly_installment'
    elif current_step == 'monthly_installment':
        return 'add_another_lender'
    
    # Handle the "add another lender" decision
    elif current_step == 'add_another_lender':
        return 'add_another_lender'  # Stay here until user answers
    
    # Handle subsequent lenders
    elif current_step == 'next_lender_name':
        return 'lender_website'
    
    # Confirmation step after all lenders are added
    elif current_step == 'confirmation':
        return 'confirmation'
    
    # Correction step
    elif current_step == 'correction_choice':
        return 'correction_choice'
    
    elif current_step == 'correction_detail':
        return 'correction_detail'
    
    elif current_step == 'complete':
        return 'complete'
    
    # Default case
    return 'complete'

def ask_question(question_type):
    """Ask the appropriate question based on type"""
    questions = {
        'name': "Selamat datang! Saya SantAI, asisten finansial Anda. Sebelum mulai, boleh saya tahu nama Anda?",
        'age': f"Terima kasih, {st.session_state.user_data['name']}! Berapa usia Anda?",
        'job': f"Usia {st.session_state.user_data['age']} tahun. Apa pekerjaan Anda saat ini?",
        'income': f"Pekerjaan sebagai {st.session_state.user_data['job']}. Berapa penghasilan bulanan Anda? (dalam Rupiah)",
        'lender_name': "Terima kasih atas informasinya. Sekarang, mari kita bahas tentang pinjaman Anda. Siapa nama pemberi pinjaman?",
        'lender_website': "Apakah pinjaman tersebut memiliki website? (Jika ada, sebutkan. Jika tidak, ketik 'Tidak')",
        'loan_amount': "Berapa total jumlah pinjaman yang Anda ambil dari lender ini? (dalam Rupiah)",
        'monthly_installment': "Berapa besar angsuran bulanan yang harus Anda bayarkan untuk lender ini? (dalam Rupiah)",
        'add_another_lender': f"Apakah Anda memiliki pinjaman dari lender lain yang ingin ditambahkan? (Jawab dengan: Ya atau Tidak)",
        'next_lender_name': "Siapa nama pemberi pinjaman lainnya?",
        'confirmation': "confirmation_special",  # Special case - handled separately
        'correction_choice': "correction_choice_special",  # Special case - handled separately
        'correction_detail': "correction_detail_special",  # Special case - handled separately
        'complete': "Terima kasih! Data semua lender telah tercatat."
    }
    return questions.get(question_type, "Terima kasih! Data Anda sudah lengkap.")

def extract_number(text):
    """Extract numbers from text input, handling various formats"""
    try:
        # Remove all non-digit characters except commas and dots
        cleaned = ''.join(c for c in text if c.isdigit() or c in [',', '.'])
        # Remove commas and convert to float, then to int
        cleaned = cleaned.replace(',', '').replace('.', '')
        return int(cleaned)
    except:
        return None

def extract_registered_lenders_from_text():
    """Extract actual lender names from the regulatory text"""
    if not vectorstore:
        return []
    
    try:
        # Get all documents from the vector store
        all_docs = vectorstore.similarity_search("pinjaman fintech lender", k=50)
        
        registered_lenders = []
        
        # Patterns to look for lender names
        patterns = [
            r'PT\.?[\s\w]+',
            r'CV\.?[\s\w]+', 
            r'[\w\s]+FINANCE',
            r'[\w\s]+FIN TECH',
            r'[\w\s]+DIGITAL',
            r'[\w\s]+TEKNOLOGI',
            r'[\w\s]+INDONESIA'
        ]
        
        for doc in all_docs:
            content = doc.page_content
            # Look for company names typically in ALL CAPS or Title Case
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                # Look for lines that might contain lender names
                if any(keyword in line.upper() for keyword in ['PT', 'CV', 'FINANCE', 'BANK', 'DIGITAL']):
                    # Clean the line - take only the first part before any punctuation
                    clean_line = re.split(r'[.,;:]', line)[0].strip()
                    if len(clean_line) > 3 and clean_line not in registered_lenders:
                        registered_lenders.append(clean_line)
        
        return list(set(registered_lenders))  # Remove duplicates
        
    except Exception as e:
        return []

def find_closest_registered_name(lender_name):
    """Find the closest registered lender name from regulatory data"""
    if not vectorstore:
        return lender_name, "TIDAK TERCATAT DALAM DAFTAR OJK - PINJAMAN ILEGAL"
    
    try:
        # Get actual registered lenders from the text
        registered_lenders = extract_registered_lenders_from_text()
        
        # Simple similarity check - you might want to use more advanced matching
        lender_name_lower = lender_name.lower()
        
        for registered in registered_lenders:
            registered_lower = registered.lower()
            # Check if lender name is similar to any registered name
            if (lender_name_lower in registered_lower or 
                registered_lower in lender_name_lower or
                fuzz.ratio(lender_name_lower, registered_lower) > 70):  # You might need to install python-Levenshtein
                return registered, "Tercatat dalam daftar OJK - PINJAMAN LEGAL"
        
        # If no match found, it's illegal
        return lender_name, "TIDAK TERCATAT DALAM DAFTAR OJK - PINJAMAN ILEGAL"
            
    except Exception as e:
        return lender_name, "TIDAK TERCATAT DALAM DAFTAR OJK - PINJAMAN ILEGAL"

# Simple fuzzy matching function since we might not have the library
def simple_similarity(str1, str2):
    """Simple similarity check without external libraries"""
    str1 = str1.lower()
    str2 = str2.lower()
    
    # Check for exact substring match
    if str1 in str2 or str2 in str1:
        return True
    
    # Check for word overlap
    words1 = set(str1.split())
    words2 = set(str2.split())
    common_words = words1.intersection(words2)
    
    if len(common_words) >= 1:  # At least one common word
        return True
    
    return False

def generate_confirmation_message():
    """Generate the confirmation message with corrected lender names from regulatory data"""
    user_info = st.session_state.user_data
    
    # Build confirmation message
    confirmation_msg = "## 📋 Konfirmasi Daftar Pinjaman Anda\n\n"
    confirmation_msg += "Berikut adalah daftar pinjaman Anda berdasarkan data regulasi OJK:\n\n"
    
    for i, lender in enumerate(user_info['lenders'], 1):
        original_name = lender['name']
        corrected_name, status = find_closest_registered_name(original_name)
        
        # Use the corrected name for display (only if it's different and legal)
        display_name = corrected_name if "LEGAL" in status else original_name
        
        confirmation_msg += f"**Pinjaman {i}:**\n"
        confirmation_msg += f"- **Nama Pemberi Pinjaman:** {display_name}\n"
        confirmation_msg += f"- **Status:** {status}\n"
        confirmation_msg += f"- **Jumlah Pinjaman:** Rp {lender.get('loan_amount', 0):,}\n"
        confirmation_msg += f"- **Angsuran Bulanan:** Rp {lender.get('monthly_installment', 0):,}\n\n"
    
    confirmation_msg += "**Apakah daftar di atas sudah sesuai?**\n"
    confirmation_msg += "- Ketik **'Sesuai'** jika data sudah benar\n"
    confirmation_msg += "- Ketik **'Tidak Sesuai'** jika ada yang perlu diperbaiki"
    
    return confirmation_msg

def generate_correction_choice_message():
    """Generate message for choosing which lender to correct"""
    user_info = st.session_state.user_data
    correction_msg = "**Pinjaman mana yang ingin Anda perbaiki?**\n\n"
    
    for i, lender in enumerate(user_info['lenders'], 1):
        correction_msg += f"**{i}. {lender['name']}** - Rp {lender.get('loan_amount', 0):,}/bulan Rp {lender.get('monthly_installment', 0):,}\n"
    
    correction_msg += f"\n**{len(user_info['lenders']) + 1}. Kembali ke konfirmasi**\n\n"
    correction_msg += "Silakan ketik nomor pilihan Anda:"
    
    return correction_msg

def generate_correction_detail_message():
    """Generate message for detailed correction"""
    lender_index = st.session_state.lender_to_correct
    lender = st.session_state.user_data['lenders'][lender_index-1]
    
    correction_msg = f"## ✏️ Perbaikan Data Pinjaman {lender_index}\n\n"
    correction_msg += f"**Data saat ini:**\n"
    correction_msg += f"- Nama Lender: {lender['name']}\n"
    correction_msg += f"- Jumlah Pinjaman: Rp {lender.get('loan_amount', 0):,}\n"
    correction_msg += f"- Angsuran Bulanan: Rp {lender.get('monthly_installment', 0):,}\n\n"
    correction_msg += "**Silakan berikan data yang baru:**\n"
    correction_msg += "Format: `Nama Lender, Jumlah Pinjaman, Angsuran Bulanan`\n"
    correction_msg += "Contoh: `PT ABC Finance, 10000000, 500000`\n\n"
    correction_msg += "Anda bisa mengisi semua data atau hanya yang ingin diubah:"
    
    return correction_msg

def process_detailed_correction(user_input, lender_index):
    """Process detailed correction input"""
    try:
        parts = [part.strip() for part in user_input.split(',')]
        lender = st.session_state.user_data['lenders'][lender_index-1]
        
        updated_fields = []
        
        # Update lender name if provided
        if len(parts) >= 1 and parts[0]:
            lender['name'] = parts[0]
            updated_fields.append("nama lender")
        
        # Update loan amount if provided
        if len(parts) >= 2 and parts[1]:
            loan_amount = extract_number(parts[1])
            if loan_amount is not None:
                lender['loan_amount'] = loan_amount
                updated_fields.append("jumlah pinjaman")
        
        # Update monthly installment if provided
        if len(parts) >= 3 and parts[2]:
            installment = extract_number(parts[2])
            if installment is not None:
                lender['monthly_installment'] = installment
                updated_fields.append("angsuran bulanan")
        
        if updated_fields:
            return f"✅ Berhasil memperbarui {', '.join(updated_fields)} untuk pinjaman {lender_index}!"
        else:
            return "❌ Tidak ada data yang diperbarui. Pastikan format sesuai contoh."
            
    except Exception as e:
        return f"❌ Error dalam memproses data: {str(e)}"


# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Check if all data is complete AND we haven't shown analysis yet
if st.session_state.user_data['data_complete'] and not st.session_state.get('analysis_shown', False):
    
    # Set flag immediately to prevent re-running
    st.session_state.analysis_shown = True
    
    # Store the analyzing message in session state so it persists across reruns
    st.session_state.analyzing_msg = "🔍 Terima kasih! Semua data Anda telah tercatat. Saya akan menganalisis kondisi finansial Anda. dan merujuk ke regulasi OJK..."
    st.session_state.messages.append({"role": "assistant", "content": st.session_state.analyzing_msg})
    
    # Force a rerun to generate the analysis
    st.session_state.should_generate_analysis = True
    st.rerun()

# Generate analysis after rerun
if st.session_state.get('should_generate_analysis', False):
    st.session_state.should_generate_analysis = False  # Reset flag
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Prepare user data for analysis - handle multiple lenders
        user_info = st.session_state.user_data
        total_monthly_installment = sum(lender.get('monthly_installment', 0) for lender in user_info['lenders'])
        total_loan_amount = sum(lender.get('loan_amount', 0) for lender in user_info['lenders'])
        debt_to_income_ratio = (total_monthly_installment / user_info['income']) * 100 if user_info['income'] > 0 else 0
        
        # Build context for all lenders from knowledge base with legal status
        lenders_context = ""
        illegal_lenders = []
        
        for i, lender in enumerate(user_info['lenders'], 1):
            lender_name = lender['name']
            corrected_name, status = find_closest_registered_name(lender_name)
            
            # Check if this is an illegal lender
            if "ILEGAL" in status:
                illegal_lenders.append(lender_name)
                lenders_context += f"\n\n**Lender {i} - {lender_name}:** ILEGAL - TIDAK TERCATAT DALAM OJK"
            else:
                docs = vectorstore.similarity_search(lender_name, k=2)
                context_content = "\n".join([doc.page_content for doc in docs])
                lenders_context += f"\n\n**Lender {i} - {lender_name}:** LEGAL - TERCATAT OJK\n{context_content}"
        
        # Create comprehensive analysis prompt with emphasis on illegal lenders
        analysis_prompt = f"""ANDA ADALAH SANTAI - AHLI ANALIS FINANSIAL YANG EMPATIK. GUNAKAN HANYA INFORMASI DARI DOKUMEN REGULASI OJK YANG DISEDIAKAN.

**DATA PENGGUNA:**
- Nama: {user_info['name']}
- Usia: {user_info['age']} tahun
- Pekerjaan: {user_info['job']}
- Penghasilan: Rp {user_info['income']:,.0f}/bulan
- Total Pinjaman: Rp {total_loan_amount:,.1f}
- Total Angsuran: Rp {total_monthly_installment:,.0f}/bulan
- Rasio DSR: {debt_to_income_ratio:.1f}%

**LENDER:**
""" + "\n".join([f"- {lender['name']}: Rp {lender.get('loan_amount', 0):,.0f} ({find_closest_registered_name(lender['name'])[1]})" 
                for lender in user_info['lenders']]) + f"""

**PINJAMAN ILEGAL YANG TIDAK PERLU DIBAYAR:**
{', '.join(illegal_lenders) if illegal_lenders else 'Tidak ada pinjaman ilegal'}

**REFERENSI REGULASI OJK:**
{lenders_context}

**TUGAS PENTING:**
1. Beri tahu user dengan JELAS mana pinjaman LEGAL dan ILEGAL
2. Untuk pinjaman ILEGAL: tekankan bahwa TIDAK PERLU DIBAYAR dan beri langkah melapor
3. Untuk pinjaman LEGAL: beri analisis kemampuan bayar
4. Berikan rekomendasi konkret berdasarkan status legalitas
5. Gunakan bahasa yang menenangkan tapi tegas untuk pinjaman ilegal

**HASIL ANALISIS:**
"""
        
        # Generate analysis
        try:
            for chunk in llm.stream(analysis_prompt):
                full_response += chunk.content
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
            
            # Replace the analyzing message with actual analysis using session state
            if (st.session_state.messages and 
                st.session_state.get('analyzing_msg') and 
                st.session_state.messages[-1]["content"] == st.session_state.analyzing_msg):
                st.session_state.messages[-1] = {"role": "assistant", "content": full_response}
            else:
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            error_msg = f"❌ Maaf, terjadi error dalam menganalisis data: {str(e)}"
            message_placeholder.markdown(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Show restart button after analysis is shown
if st.session_state.get('analysis_shown', False):
    if st.button("🔄 Mulai Konsultasi Baru", key="restart_btn"):
        # Reset everything
        st.session_state.user_data = {
            'name': None, 'age': None, 'job': None, 'income': None,
            'lenders': [],  # List to store multiple lenders
            'data_complete': False
        }
        st.session_state.current_step = 'name'
        st.session_state.analysis_shown = False
        st.session_state.should_generate_analysis = False
        st.session_state.analyzing_msg = None
        st.session_state.current_lender = {}
        st.session_state.add_another_lender = None
        st.session_state.confirmation_step = False
        st.session_state.correcting_lender = False
        st.session_state.lender_to_correct = None
        st.session_state.messages = []
        st.rerun()

elif llm:  # Only show input if LLM is available
    # Show current question
    current_question_type = st.session_state.current_step
    current_question = ask_question(current_question_type)
    
    # Handle special cases for confirmation and correction steps
    if current_question == "confirmation_special":
        current_question = generate_confirmation_message()
    elif current_question == "correction_choice_special":
        current_question = generate_correction_choice_message()
    elif current_question == "correction_detail_special":
        current_question = generate_correction_detail_message()
    
    if not st.session_state.user_data['data_complete']:
        if not st.session_state.messages or st.session_state.messages[-1]["role"] != "assistant" or st.session_state.messages[-1]["content"] != current_question:
            st.session_state.messages.append({"role": "assistant", "content": current_question})
            with st.chat_message("assistant"):
                st.markdown(current_question)

    # Handle user input
    if prompt := st.chat_input("Jawaban Anda..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Process the answer based on current step
        current_step = st.session_state.current_step
        
        if current_step == 'name':
            st.session_state.user_data['name'] = prompt
            st.session_state.current_step = get_next_question()
            
        elif current_step == 'age':
            try:
                st.session_state.user_data['age'] = int(prompt)
                st.session_state.current_step = get_next_question()
            except ValueError:
                st.error("Mohon masukkan usia dalam angka")
                st.session_state.messages.append({"role": "assistant", "content": "Mohon masukkan usia Anda dalam angka (contoh: 25)"})
                
        elif current_step == 'job':
            st.session_state.user_data['job'] = prompt
            st.session_state.current_step = get_next_question()
            
        elif current_step == 'income':
            try:
                income = extract_number(prompt)
                if income is not None:
                    st.session_state.user_data['income'] = income
                    st.session_state.current_step = get_next_question()
                else:
                    st.error("Mohon masukkan angka yang valid")
                    st.session_state.messages.append({"role": "assistant", "content": "Mohon masukkan penghasilan dalam format angka (contoh: 5000000 atau Rp 5.000.000)"})
            except:
                st.error("Mohon masukkan angka yang valid")
                st.session_state.messages.append({"role": "assistant", "content": "Mohon masukkan penghasilan dalam format angka (contoh: 5000000 atau Rp 5.000.000)"})
                
        elif current_step == 'lender_name':
            st.session_state.current_lender = {'name': prompt}
            st.session_state.current_step = get_next_question()
            
        elif current_step == 'lender_website':
            st.session_state.current_lender['website'] = prompt if prompt.lower() != 'tidak' else None
            st.session_state.current_step = get_next_question()
            
        elif current_step == 'loan_amount':
            try:
                loan_amount = extract_number(prompt)
                if loan_amount is not None:
                    st.session_state.current_lender['loan_amount'] = loan_amount
                    st.session_state.current_step = get_next_question()
                else:
                    st.error("Mohon masukkan angka yang valid")
                    st.session_state.messages.append({"role": "assistant", "content": "Mohon masukkan jumlah pinjaman dalam format angka (contoh: 10000000 atau Rp 10.000.000)"})
            except:
                st.error("Mohon masukkan angka yang valid")
                st.session_state.messages.append({"role": "assistant", "content": "Mohon masukkan jumlah pinjaman dalam format angka (contoh: 10000000 atau Rp 10.000.000)"})
                
        elif current_step == 'monthly_installment':
            try:
                installment = extract_number(prompt)
                if installment is not None:
                    st.session_state.current_lender['monthly_installment'] = installment
                    
                    # Add completed lender to the list
                    st.session_state.user_data['lenders'].append(st.session_state.current_lender.copy())
                    
                    # Ask if user wants to add another lender
                    st.session_state.current_step = 'add_another_lender'
                else:
                    st.error("Mohon masukkan angka yang valid")
                    st.session_state.messages.append({"role": "assistant", "content": "Mohon masukkan jumlah angsuran dalam format angka (contoh: 1500000 atau Rp 1.500.000)"})
            except:
                st.error("Mohon masukkan angka yang valid")
                st.session_state.messages.append({"role": "assistant", "content": "Mohon masukkan jumlah angsuran dalam format angka (contoh: 1500000 atau Rp 1.500.000)"})
                
        elif current_step == 'add_another_lender':
            if prompt.lower() in ['ya', 'yes', 'y', 'iya', 'iye']:
                st.session_state.current_step = 'next_lender_name'
            elif prompt.lower() in ['tidak', 'no', 'n', 'ga', 'gak']:
                # Move to confirmation step
                st.session_state.current_step = 'confirmation'
                st.session_state.confirmation_step = True
            else:
                st.error("Mohon jawab dengan Ya atau Tidak")
                st.session_state.messages.append({"role": "assistant", "content": "Mohon jawab dengan Ya atau Tidak"})
                
        elif current_step == 'next_lender_name':
            st.session_state.current_lender = {'name': prompt}
            st.session_state.current_step = 'lender_website'
            
        elif current_step == 'confirmation':
            # Handle confirmation responses
            if prompt.lower() in ['sesuai', 'confirm', 'ya', 'yes', 'benar']:
                st.session_state.user_data['data_complete'] = True
                st.session_state.current_step = 'complete'
                st.session_state.confirmation_step = False
            elif prompt.lower() in ['tidak sesuai', 'tidak', 'no', 'revisi', 'ubah']:
                st.session_state.current_step = 'correction_choice'
            else:
                st.error("Mohon ketik 'Sesuai' atau 'Tidak Sesuai'")
                st.session_state.messages.append({"role": "assistant", "content": "Mohon ketik 'Sesuai' jika data benar, atau 'Tidak Sesuai' jika perlu diperbaiki"})
                
        elif current_step == 'correction_choice':
            try:
                choice = int(prompt)
                num_lenders = len(st.session_state.user_data['lenders'])
                
                if 1 <= choice <= num_lenders:
                    # User wants to correct a specific lender
                    st.session_state.lender_to_correct = choice
                    st.session_state.current_step = 'correction_detail'
                elif choice == num_lenders + 1:
                    # Back to confirmation
                    st.session_state.current_step = 'confirmation'
                else:
                    st.error("Pilihan tidak valid")
                    st.session_state.messages.append({"role": "assistant", "content": "Mohon pilih nomor yang sesuai dengan pilihan yang tersedia."})
            except ValueError:
                st.error("Mohon masukkan angka")
                st.session_state.messages.append({"role": "assistant", "content": "Mohon masukkan nomor pilihan Anda (contoh: 1, 2, 3)"})
                
        elif current_step == 'correction_detail':
            # Process detailed correction
            result_msg = process_detailed_correction(prompt, st.session_state.lender_to_correct)
            st.session_state.messages.append({"role": "assistant", "content": result_msg})
            
            # Go back to confirmation to show updated list
            st.session_state.current_step = 'confirmation'
            st.session_state.lender_to_correct = None

        st.rerun()