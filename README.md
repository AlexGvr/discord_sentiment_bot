# Discord sentiment bot (Russian)

---

This bot allows you to count and separate the previously sent messages of users by sentiment.

---
### Info
RuBert (https://huggingface.co/DeepPavlov/rubert-base-cased) was taken as a predictive model (predict three classes).
Fine-tuned on RuSentiment dataset for sentiment analysis of Russian social media (https://github.com/strawberrypie/rusentiment)
with ~80% accuracy on validation set.

---
### Setup:
* Firstly clone to your local folder where you wanted to store project:
```git clone https://github.com/AlexGvr/discord_sentiment_bot.git your_folder ```
* Next, you will be required python 3 installed on your computer
* Then you will be required install all needed packages:
```pip install -r requirements.txt```
* In config.py specify the name of the bot you want and bot token
* Last, run discord_bot.py
---
### Functionality:

Bot supports multiple commands

* !hello - greats you
  
  ![image](https://user-images.githubusercontent.com/39123866/116409539-9bc7d180-a83c-11eb-8ac5-72ce0489ea9a.png)
  
* !joined - give information when a member joined the channel
![image](https://user-images.githubusercontent.com/39123866/116418512-e3525b80-a844-11eb-93b2-2c39be6b7fea.png)
    
* !youtube - display the first result on YouTube of the request
![image](https://user-images.githubusercontent.com/39123866/116418662-0846ce80-a845-11eb-970d-ed737bdbaa42.png)

* =fetch - calculate count of negative, neutral and positive messages of a user
![image](https://user-images.githubusercontent.com/39123866/116418310-b1d99000-a844-11eb-9b10-1d8a15da3321.png)

* !stats - show results of sentiment analysis of all participants who used the command =fetch
![image](https://user-images.githubusercontent.com/39123866/116418863-30cec880-a845-11eb-94f0-821a1eb21089.png)

* !toxic - defined negative message or not

  ![image](https://user-images.githubusercontent.com/39123866/116419065-5a87ef80-a845-11eb-97e5-d008f323da1f.png)


