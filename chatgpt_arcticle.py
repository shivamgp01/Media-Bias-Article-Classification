import os
import openai
import sys
import csv
import time

def write_line(original_score,article,gpt_score,raw_gpt):
	'''
	Appends the origianl score, article, gpt score, and the raw output to the file
	'''
	f = open("gpt_out.csv", "a",encoding='utf-8')
	to_write = '\n' + original_score + ',' + gpt_score + ',' + raw_gpt + ',' + article[:60]
	f.write(to_write)
	f.close()

def ask_chat_gpt(article):
	'''
	Uses chatgpt api to get a solution and return the score as well as raw reply
	'''
	messages = [
        {"role": "system", "content": "You are a helpful assistant."},
	]
	openai.api_key = 'YOUR_API_KEY'

	message = 'Is this article politically left or right? Please just say Left, Neutral, or Right:' + article[:4000]
	messages.append(
	{"role": "user", "content": message},
	)
	chat = openai.ChatCompletion.create(
		model="gpt-3.5-turbo", messages=messages
	)
	reply = chat.choices[0].message.content
	messages.append({"role": "assistant", "content": reply})

	gpt_score = '-1'
	if 'Left' in reply:
		gpt_score = '0'
	elif 'Neutral' in reply:
		gpt_score = '1'
	elif 'Right' in  reply: 
		gpt_score = '2'
	return gpt_score,reply

def open_and_scan_file():

	csvfile = open('dataset_all.csv',encoding='utf-8')
	reader = csv.DictReader(csvfile)
	count = 0
	actual_count = 0
	num_correct = 0

	for row in reader:
		original_score = row['\ufefflabel']
		article = row['text']
		if article.startswith('HuffPost'):
			gpt_score = '-1'
			reply = 'HuffPost Article'
			write_line(original_score,article,'-1','HuffPost Article')
		else:
			gpt_score, reply = ask_chat_gpt(article)
			time.sleep(20.5)
			write_line(original_score,article,gpt_score,reply)
			actual_count += 1
			if original_score == gpt_score:
				num_correct += 1
		count += 1
		print(count)
		print(original_score + ' ' + gpt_score + ' ' + reply)

	print(num_correct)
	print(actual_count)
	accuracy = num_correct / actual_count
	print('accuracy: ' + str(accuracy))

def test_api():
	messages = [
        {"role": "system", "content": "You are a helpful assistant."},
	]
    
    # Value from https://platform.openai.com/account/org-settings, deleted for privacy
	openai.api_key = os.getenv("OPENAI_API_KEY")

	while True:
		message = input("User : ")
		if message:
			messages.append(
			{"role": "user", "content": message},
			)
			chat = openai.ChatCompletion.create(
				model="gpt-3.5-turbo", messages=messages
			)
	
		reply = chat.choices[0].message.content
		print(f"ChatGPT: {reply}")
		messages.append({"role": "assistant", "content": reply})


if __name__ == '__main__':
	open_and_scan_file()
