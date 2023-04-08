
<h1 align="center">
  <br>
  <img src="https://img.freepik.com/free-vector/cute-artificial-intelligence-robot-isometric-icon_1284-63045.jpg" width="200">
  <br>
  Docai
  <br>
</h1>

<h4 align="center">Docai is a GPT-3 based Question Answering System that can provide answers based on a PDF, DOCX, and TXT files. </h4>

<p align="center">
  <a href="">
    <img src="https://img.shields.io/badge/os-windows-blue.svg?maxAge=2592000&amp;style=flat"
         >
  </a>
  <a href=""><img src="https://img.shields.io/badge/version-1.0-red.svg?maxAge=2592000&amp;style=flat"></a>
</p>

<p align="center">
  <a href="#key-features">Key Features</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#Requirements">Requirements</a> •
  <a href="#Copyright">Copyright</a>
</p>

<p align="center">
<a href=""><img src="https://i.giphy.com/media/HdjEnj3U6b6hGzcRsW/giphy.webp"></a>
</p>

## Key Features

* File handling
  - The script supports PDF, DOCX, and TXT files
  - Read the content using the pdfplumber, docx, and built-in open() functions
* GPT-3 integration
  - The script uses the OpenAI GPT-3 model, specifically the text-davinci-003 engine, to generate answers to questions.
* Confidence scoring
  - The script calculates confidence scores for the generated answers using log probabilities returned by the GPT-3 API.
* Concurrency
  - It uses the concurrent.futures.ThreadPoolExecutor to process questions concurrently, potentially speeding up the process.
* Text preprocessing
  - The script splits the input document into chunks to fit within GPT-3's token limit, and post-processes the answers to remove duplicate sentences.
* Saving conversation history
  - The script allows users to save the conversation history to a text file.
* Gui
  - The script provides a friendly graphical user interface built using the tkinter library and ttkthemes allowing users to select a file, input a question, view the answer, and save the conversation history.

## How To Use

- Put you api key in line 52
- Run the script
- Select your file
- Enter your question and click submit

It's as simple as that

> **Note**
> We will provide an executable version soon

## Requirements

* pip install colorama openai pdfplumber python-docx

## Copyright

All rights reserved to Bropocalypse Team.
