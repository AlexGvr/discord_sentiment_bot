import discord
from discord.ext import commands
from config import settings
import re
from urllib import parse, request
from transformers import BertTokenizer, BertModel,BertConfig
import numpy as np
import os
import random
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from collections import Counter
import pandas as pd

global points
points = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bot = commands.Bot(command_prefix = settings['prefix'])
client = discord.Client()


@bot.command()
async def hello(ctx):
    """
    Function that greets you with a command !hello,
    the greeting is chosen randomly from the list, with a fixed probability
    :param ctx:
    """
    author = ctx.message.author
    greetings = [
        'Привет малыш',
        'Привет малютка',
        'Привет красавчик',
        'Привет зайка',
        'Привет пидрила',
        'Здарова хуила'
    ]
    weights = [10,10,10,10,1,1]
    say = random.choices(greetings,weights, k=1)[0]
#     print(ctx)
    await ctx.send(f'{say}, {author.mention}!')

@bot.command()
async def joined(ctx, *, member: discord.Member):
    """
    Function that give information when a member joined the channel with a command !joined
    :param ctx:
    :param member:
    """
    try:
        await ctx.send('{0} joined on {0.joined_at}'.format(member))
    except:
        await ctx.send('{0} нет в этом чате долбаёб'.format(member))


@bot.command()
async def youtube(ctx, *, search):
    """
    Function that displays the first result on YouTube of the request after the command !youtube
    :param ctx:
    :param search:
    """
    query_string = parse.urlencode({'search_query': search})
    html_content = request.urlopen('http://www.youtube.com/results?' + query_string)
    search_results = re.findall(r"watch\?v=(\S{11})", html_content.read().decode())
    # I will put just the first result, you can loop the response to show more results
    await ctx.send('https://www.youtube.com/watch?v=' + search_results[0])

@bot.event
async def on_message(ctx):
        """
        Function that calculates count of negative, neutral and positive messages of a user after the command =fetch
        :param ctx:
        :return:
        """
        if ctx.content.startswith('=fetch'):
            comments = []
            fetch_target = ctx.mentions[0] if ctx.mentions else ctx.author
            dp_name = str(fetch_target)[:-5]
            embed = discord.Embed(title="I'm busy counting~", color=0x9975b9)
            calc = await ctx.channel.send(embed=embed)
            counter = 0
            Ctotal = 0
            async for msg in (discord.abc.Messageable.history(ctx.channel,limit=5000)):

                Ctotal += 1
                if msg.author == fetch_target:
                    q = ('!', 'www.', 'http', '=')
                    if msg.content.startswith(q) == False:

                        comments.append(msg.content)
                        counter += 1

                sum = Ctotal
                sum2 = counter * 100/ sum
                sum3 = round(sum2, 2)
                if (Ctotal % 5000 == 0):
                    embed2 = discord.Embed(title="I'm busy counting~", color=0xff0000)
                    embed2.set_author(name=dp_name, icon_url=fetch_target.avatar_url)
                    embed2.add_field(name="Total messages counted:", value="{} messages so far".format(Ctotal),
                                     inline=False)
                    embed2.add_field(name="Messages of {}:".format(dp_name), value="{} messages".format(counter),
                                     inline=False)
                    embed2.add_field(name="Your message participation percentage:", value="{}%".format(sum3), inline=False)
                    embed2.set_footer(text="counting done soonTM")
                    await bot.edit_message(calc, embed=embed2)

            # print(comments)
            q = toxic_check_1(comments)
            toxic_count = q[0]
            neutral_count = q[1]
            positive_count = q[2]
            embed3 = discord.Embed(title="I'm done counting!", color=0x00ff00)
            embed3.set_author(name=dp_name, icon_url=fetch_target.avatar_url)
            embed3.add_field(name="Total messages counted:", value="{} messages".format(Ctotal), inline=False)
            embed3.add_field(name="Messages of {}:".format(dp_name), value="{} messages {}% of all".format(counter, sum3), inline=False)
            embed3.add_field(name="Toxic messages of {}:".format(dp_name),
                             value="{} toxic messages {}% of all".format(toxic_count, np.round(toxic_count*100/counter,2)), inline=False)
            embed3.add_field(name="Neutral messages of {}:".format(dp_name),
                             value="{} neutral messages {}% of all".format(neutral_count, np.round(neutral_count*100/counter,2)), inline=False)
            embed3.add_field(name="Positive messages of {}:".format(dp_name),
                             value="{} positive messages {}% of all".format(positive_count, np.round(positive_count*100/counter,2)), inline=False)
            embed3.set_footer(text="counting done!")
            await discord.Message.delete(calc)
            await ctx.channel.send( embed=embed3)
            await ctx.channel.send(ctx.author.mention)

            df = pd.DataFrame()
            df['User'] = [dp_name]
            df['Total_messages'] = [counter]
            df['Toxic_messages'] = [toxic_count]
            df['Toxic_messages_percent'] = [np.round(toxic_count*100/counter,2)]
            df['Neutral_messages'] = [neutral_count]
            df['Neutral_messages_percent'] = [np.round(neutral_count*100/counter,2)]
            df['Positive_messages'] = [positive_count]
            df['Positive_messages_percent'] = [np.round(positive_count*100/counter,2)]
            df.to_csv('stats_'+f'{dp_name}'+'.csv',index = False)
        await bot.process_commands(ctx)
        return

@bot.command()
async def echo(ctx, *, content:str):
    """
    Function that duplicates the messageб this command !echo
    :param ctx:
    :param content:
    :return:
    """
    await ctx.send(content)
    return

@bot.command()
async def stats(ctx):
    """
    Function that shows results of sentiment analysis of all participants who used the command =fetch,
    to see stats use command !stats
    :param ctx:
    :return:
    """
    files = []
    for file in os.listdir(os.getcwd()):
        if file.endswith('.csv'):
            files.append(file)

    for i,k in enumerate(files):
        if i == 0:
            df = pd.read_csv(k)
        else:
            df = pd.concat((df,pd.read_csv(k)))

    embed = discord.Embed(title=f"__**{ctx.guild.name} Results:**__", color=0x00ff00)
    df = df.reset_index(drop = True)
    for i in range(df.shape[0]):  # process embed
        embed.add_field(name=f"**{df['User'][i]}**",
                        value=f'Total_messages: {df["Total_messages"][i]}\nToxic_messages: {df["Toxic_messages"][i]}\n'
                              f'Toxic_messages_percent: {df["Toxic_messages_percent"][i]}\n'
                              f'Neutral_messages: {df["Neutral_messages"][i]}\n'
                              f'Neutral_messages_percent: {df["Neutral_messages_percent"][i]}\n'
                              f'Positive_messages: {df["Positive_messages"][i]}\n'
                              f'Positive_messages_percent: {df["Positive_messages_percent"][i]}',
                        inline=True)
    df = df.sort_values(by='Toxic_messages_percent', ascending=False)
    df = df.reset_index(drop = True)
    embed.set_image(url='https://wmpics.pics/di-3GOE.png')
    embed.add_field(name="The most toxic man of the channel is {}".format(df['User'][0]),\
                    value= f'Percent of toxic messages is {df["Toxic_messages_percent"][0]}', inline=False)
    await ctx.send(embed=embed)
    return



@bot.command()
async def toxic(ctx, *, content:str):
    """
    Function that defined negative message or not< with command !toxic
    :param ctx:
    :param content:
    :return:
    """
    model, tokenizer = download_model()
    tokens = [
        tokenizer.convert_tokens_to_ids(
            ['[CLS]'] + tokenizer.tokenize(t) + ['[SEP]']
        )
        for t in [content]]

    tokens = pad_sequences(
        tokens,
        maxlen=100,
        dtype="long",
        truncating="post",
        padding="post"
    )

    attention_masks = [[float(i > 0) for i in seq] for seq in tokens]
    prediction_inputs = torch.tensor(tokens).type(torch.LongTensor)
    prediction_masks = torch.tensor(attention_masks).type(torch.LongTensor)

    prediction_data = TensorDataset(
        prediction_inputs,
        prediction_masks
    )
    # print('Hello')
    prediction_dataloader = DataLoader(
        prediction_data,
        sampler=SequentialSampler(prediction_data),
        batch_size=32
    )

    test_preds = []

    for batch in (prediction_dataloader):

        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask = batch

        with torch.no_grad():
            logits = model(b_input_ids, attention_mask=b_input_mask)


            logits = logits.detach().cpu().numpy()
            # print(logits)
            batch_preds = np.argmax(logits, axis=1)
            test_preds.extend(batch_preds)

    # print(batch_preds)
    if batch_preds[0] == 0:
        return await ctx.channel.send('Toxic comment')
    else:
        return await ctx.channel.send('Not toxic comment')

def download_model():
    """
    Download Bert model
    :return:
    """
    SEED = 1234

    random.seed(SEED)
    torch.manual_seed(SEED)
    # torch.backends.cudnn.deterministic = True


    if device == 'cpu':
        print('cpu')
    else:
        n_gpu = torch.cuda.device_count()
        print(torch.cuda.get_device_name(0))

    print(os.getcwd())
    config = BertConfig.from_pretrained(os.getcwd() + '/config.json')

    class MyBert(torch.nn.Module):
        def __init__(self, config):
            super(MyBert, self).__init__()
            self.h1 = BertModel(config=config)
            self.classifier = torch.nn.Linear(768, 3)

        def forward(self, input_ids, attention_mask):
            output_1 = self.h1(input_ids=input_ids, attention_mask=attention_mask)
            hidden_state = output_1[0]
            pooler = hidden_state[:, 0]
            output = self.classifier(pooler)
            return output

    model = MyBert(config)
    model.load_state_dict(torch.load(os.getcwd() + '/best_chekpoint.pt'))

    model.to(device)
    model.eval()
    tokenizer = BertTokenizer.from_pretrained(os.getcwd())

    return model, tokenizer

def toxic_check_1(comments: list):

    model, tokenizer = download_model()
    tokens = [
        tokenizer.convert_tokens_to_ids(
            ['[CLS]'] + tokenizer.tokenize(t) + ['[SEP]']
        )
        for t in comments]

    tokens = pad_sequences(
        tokens,
        maxlen=100,
        dtype="long",
        truncating="post",
        padding="post"
    )

    attention_masks = [[float(i > 0) for i in seq] for seq in tokens]
    prediction_inputs = torch.tensor(tokens).type(torch.LongTensor)
    prediction_masks = torch.tensor(attention_masks).type(torch.LongTensor)

    prediction_data = TensorDataset(
        prediction_inputs,
        prediction_masks
    )
    # print('Helo')
    prediction_dataloader = DataLoader(
        prediction_data,
        sampler=SequentialSampler(prediction_data),
        batch_size=32
    )

    test_preds = []

    for batch in (prediction_dataloader):

        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask = batch

        with torch.no_grad():
            logits = model(b_input_ids, attention_mask=b_input_mask)


            logits = logits.detach().cpu().numpy()
            # print(logits)
            batch_preds = np.argmax(logits, axis=1)
            test_preds.extend(batch_preds)

    # print(batch_preds)
    return Counter(test_preds)



bot.run(settings['token'])



