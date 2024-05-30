from openai import OpenAI
import pandas as pd
import pyreadstat

#API parameters
client = OpenAI(api_key='xxx', organization='yyy')
GPT_MODEL = "gpt-3.5-turbo-0125"


messages = [
    {"role": "system", "content": "You are a participant in a study concerning the Big Five personality traits. Acquiesce to the instructions to the best of your ability. Answer the given statements with either a 1, 2, 3, 4 or 5. Where 1 is very inaccurate, 2 is moderately inaccurate, 3 is neutral, 4 is moderately accurate and 5 is very accurate. Structure your response by separating each answer with only a comma and no space. Provide an answer for every statement. There are exactly 50 statements so give exactly 50 answers, structured as instructed. Do not end with a period."},
    {"role": "user", "content": "1, I am the life of the party. 2, I feel little concern for others. 3, I am always prepared. 4, I get stressed out easily. 5, I have a rich vocabulary. 6, I don't talk a lot. 7, I am interested in people. 8, I leave my belongings around. 9, I am relaxed most of the time. 10, I have difficulty understanding abstract ideas. 11, I feel comfortable around people. 12, I insult people. 13, I pay attention to details. 14, I worry about things. 15, I have a vivid imagination. 16, I keep in the background. 17, I sympathise with others' feelings. 18, I make a mess of things. 19, I seldom feel blue. 20, I am not interested in abstract ideas. 21, I start conversations. 22, I am not interested in abstract ideas. 23, I get chores done right away. 24, I am easily disturbed. 25, I have excellent ideas. 26, I have little to say. 27, I have a soft heart. 28, I often forget to put things back in their proper place. 29, I get upset easily. 30, I do not have a good imagination. 31, I talk to a lot of different people at parties. 32, I am not really interested in others. 33, I like order. 34, I change my mood a lot. 35, I am quick to understand things. 36, I don't like to draw attention to myself. 37, I take time out for others. 38, I shirk my duties. 39, I have frequent mood swings. 40, I use difficult words. 41, I don't mind being the centre of attention. 42, I feel others' emotion. 43, I follow a schedule. 44, I get irritated easily. 45, I spend time reflecting on things. 46, I am quiet around strangers. 47, I make people feel at ease. 48, I am exacting in my work. 49, I often feel blue. 50, I am full of ideas."}
]

#Variables
N = 70 #Number of completions
T = 0.2 #Temperature

#Function to get responses from GPT
def get_responses(n_completions):
    completions = client.chat.completions.create(
        model=GPT_MODEL,
        messages=messages,
        temperature=T,
        max_tokens=100,
        n=n_completions
    )
    return completions.choices

#Function that creates a datafame from the completions
def create_df(completions):
    responses = [choice.message.content for choice in completions]
    df = pd.DataFrame({'Answers': responses})
    df = df['Answers'].str.split(',', expand=True)
    df.columns = [f'Answer_{i}' for i in range(len(df.columns))]
    df = df.apply(pd.to_numeric, errors='coerce')
    return df

#Executes the function for GPT completions
completions = get_responses(N)

#Incorporates the dataframe
df = create_df(completions)
#Inserts a column with the set temperature
df.insert(0, 'temperature', T)

#Write the dataframe to a SPSS file
pyreadstat.write_sav(df, "~/Documents/Data_Temperature_" + str(T) +".sav", file_label="Data_Temperature_" + str(T) +"")

#Print dataframe in terminal
print(df)
