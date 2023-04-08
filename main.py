import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from ttkthemes import ThemedTk
from tkinter import messagebox
import threading

import pdfplumber
import docx
import openai
import numpy as np
import threading
import concurrent.futures
from functools import lru_cache
from pathlib import Path
from colorama import Fore, init

init()
# Set a threshold for confidence scores
confidence_threshold = 0.3

def read_file(file_path: Path):
    if file_path.suffix == '.pdf':
        with pdfplumber.open(file_path) as pdf:
            pages_text = [(page.page_number, page.extract_text()) for page in pdf.pages]
    elif file_path.suffix == '.docx':
        doc = docx.Document(file_path)
        pages_text = []
        page_number = 1
        page_text = ""

        for paragraph in doc.paragraphs:
            if "SectionBreak" in [run.style.name for run in paragraph.runs]:
                pages_text.append((page_number, page_text))
                page_number += 1
                page_text = ""
            else:
                page_text += paragraph.text + "\n"

        # Add the last page
        pages_text.append((page_number, page_text))
    elif file_path.suffix == '.txt':
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
    else:
        raise ValueError("Unsupported file format. Please use a PDF or DOCX file.")

    return pages_text

@lru_cache(maxsize=128)
def generate_answer(prompt, page_number):
    api_key = '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'
    model = 'text-davinci-003'

    openai.api_key = api_key
    
    prompt = f"Please provide a detailed, accurate, and coherent answer to the following question in approximately 100 words: {prompt}"
    
    response = openai.Completion.create(
        engine=model,
        prompt=prompt,
        max_tokens=1500,
        n=3,
        stop=None,
        temperature=0.7,  # Increase the temperature for more focused answers
        logprobs=10  # Request log probabilities
    )
    
    # Calculate confidence scores for each choice
    confidence_scores = [calculate_confidence(choice) for choice in response.choices]

    # Choose the answer with the highest confidence score
    best_choice_index = np.argmax(confidence_scores)
    best_answer = response.choices[best_choice_index].text.strip()
    best_confidence = confidence_scores[best_choice_index]

    if best_confidence < confidence_threshold:
        return None, best_confidence, page_number

    return best_answer, best_confidence, page_number

def calculate_confidence(choice):
    token_probs = choice.logprobs['token_logprobs']
    confidence = np.exp(token_probs)  # Convert log probabilities to probabilities
    average_confidence = np.mean(confidence)
    return average_confidence

def split_content(pages_text, max_tokens):
    chunks = []

    current_chunk = []
    current_chunk_tokens = 0
    current_page = None

    for page, text in pages_text:
        tokens = text.split()
        for token in tokens:
            token_length = len(token)

            if current_chunk_tokens + token_length > max_tokens:
                chunks.append((current_page, " ".join(current_chunk)))
                current_chunk = [token]
                current_chunk_tokens = token_length
            else:
                current_chunk.append(token)
                current_chunk_tokens += token_length
                current_page = page

    if current_chunk:
        chunks.append((current_page, " ".join(current_chunk)))

    return chunks

def process_question(question, content_chunk, page_number):
    prompt = f"Document content:\n{content_chunk}\n\nQuestion: {question}\nAnswer:"
    answer, confidence, page_number = generate_answer(prompt, page_number)
    return answer, confidence, page_number

def process_question_concurrently(question, content_chunks):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        tasks = [executor.submit(process_question, question, chunk, page_number) for page_number, chunk in content_chunks]
        answers_and_confidence = [task.result() for task in tasks]

    highest_confidence = max(confidence for _, confidence, _ in answers_and_confidence)

    if highest_confidence < confidence_threshold:
        return "not found", []

    concatenated_answer = " ".join(answer for answer, _, _ in answers_and_confidence)
    post_processed_answer = post_process_answer(concatenated_answer)
    page_numbers = [page_number for _, _, page_number in answers_and_confidence]

    return post_processed_answer, page_numbers

def post_process_answer(answer):
    sentences = answer.split(". ")
    unique_sentences = []
    seen_sentences = set()

    for sent in sentences:
        sent_text = sent.strip()
        if sent_text not in seen_sentences:
            unique_sentences.append(sent_text)
            seen_sentences.add(sent_text)

    return ". ".join(unique_sentences)

def main():
    file_path_str = input(f"{Fore.RED}Please enter the file path of your PDF or DOCX file: " + Fore.WHITE)
    file_path = Path(file_path_str)
    
    try:
        content = read_file(file_path)
    except ValueError as e:
        print(e)
        return

    max_tokens = 4097
    content_chunks = split_content(content, max_tokens - 20)

    conversation_history = []

    while True:
        question = input(f"{Fore.CYAN}\nMe: ")

        if question.lower() == 'exit':
            break

        conversation_history.append({"role": "system", "content": f"User asked: {question}"})

        answers_and_confidence = process_question_concurrently(question, content_chunks)
        answers, confidence_scores = zip(*answers_and_confidence)

        best_answer_index = np.argmax(confidence_scores)
        best_answer = post_process_answer(answers[best_answer_index])

        conversation_history.append({"role": "assistant", "content": f"Assistant answered: {best_answer}"})
        print(f"{Fore.GREEN}\nGPT: {best_answer}")
		
class QuestionAnsweringGUI:
    def __init__(self):
        self.root = ThemedTk(theme="plastik")
        self.root.title("GPT-3 Question Answering System")
        self.root.geometry("800x600")
        self.create_widgets()
        self.content = None
        self.max_tokens = 4097
        self.conversation_history = []

    def create_widgets(self):
        self.main_frame = ttk.Frame(self.root, padding=20)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.select_file_button = ttk.Button(self.main_frame, text="Select PDF or DOCX file", command=self.select_file_thread)
        self.select_file_button.pack(pady=(0, 20))

        self.question_label = ttk.Label(self.main_frame, text="Enter your question:", font=("Helvetica", 12, "bold"))
        self.question_label.pack()

        self.question_entry = ttk.Entry(self.main_frame, width=80, font=("Helvetica", 12))
        self.question_entry.pack(pady=(10, 20))

        self.submit_button = ttk.Button(self.main_frame, text="Submit", command=self.submit_question)
        self.submit_button.pack(pady=(0, 10))

        self.save_conversation_button = ttk.Button(self.main_frame, text="Save Conversation", command=self.save_conversation)
        self.save_conversation_button.pack(pady=(0, 20))

        self.progressbar = ttk.Progressbar(self.main_frame, mode="indeterminate")
        self.progressbar.pack(pady=(10, 0))

        self.answer_label = ttk.Label(self.main_frame, text="Answer:", font=("Helvetica", 12, "bold"))
        self.answer_label.pack(pady=(20, 0))

        self.answer_text = tk.Text(self.main_frame, wrap=tk.WORD, width=80, height=10, font=("Helvetica", 12))
        self.answer_text.pack(fill=tk.BOTH, expand=True)

    def select_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf"), ("Word files", "*.docx"), ("Text files", "*.txt")])
        if file_path:
            self.content = read_file(Path(file_path))
            self.content_chunks = split_content(self.content, self.max_tokens - 20)
            self.select_file_button["state"] = "normal"

    def select_file_thread(self):
        self.select_file_button["state"] = "disabled"
        threading.Thread(target=self.select_file).start()

    def submit_question(self):
        question = self.question_entry.get().strip()
        if question and self.content:
            self.submit_button["state"] = "disabled"
            self.progressbar.start()
            self.conversation_history.append({"role": "system", "content": question})
            threading.Thread(target=self.process_and_display_answer, args=(question,)).start()

    def process_and_display_answer(self, question):
        answer, page_numbers = process_question_concurrently(question, self.content_chunks)

        answer_with_page_numbers = f"{answer}\n(Page numbers: {', '.join(map(str, page_numbers))})"
        self.answer_text.delete(1.0, tk.END)
        self.answer_text.insert(tk.END, answer_with_page_numbers)

        self.submit_button["state"] = "normal"
        self.progressbar.stop()
        
    def save_conversation(self):
        if not self.conversation_history:
            messagebox.showinfo("Info", "No conversation history to save.")
            return

        file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
        if file_path:
            with open(file_path, "w", encoding="utf-8") as file:
                for entry in self.conversation_history:
                    role = "You" if entry["role"] == "system" else "GPT"
                    content = entry['content'].replace("User asked: ", "").replace("Assistant answered: ", "")
                    file.write(f"{role}: {content}\n")
            messagebox.showinfo("Info", "Conversation history saved successfully.")



    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    gui = QuestionAnsweringGUI()
    gui.run()