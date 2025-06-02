from dotenv import load_dotenv
load_dotenv()

import os
import random
import json
from openai import OpenAI
from openai.types.chat import ChatCompletionMessage
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

def get_room_temp():
    return str(random.randint(60, 80))

def set_room_temp(temp):
    return "DONE"

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_room_temp",
            "description": "華氏で部屋の温度を取得"
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_room_temp",
            "description": "華氏で部屋の温度を設定",
            "parameters": {
                "type": "object",
                "properties": {
                    "temp": {
                        "type": "integer",
                        "description": "華氏での望ましい部屋の温度",
                    },
                },
                "required": ["temp"],
            }
        },
    },
]

available_functions = {
    "get_room_temp": get_room_temp,
    "set_room_temp": set_room_temp
}

def process_messages(client, messages):
    # ステップ１：モデルに、ツール定義とともにメッセージを送信
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools,
    )
    response_message = response.choices[0].message

    # ステップ2:会話にモデルのレスポンスを追加
    # （関数呼び出し、または通常のメッセージの可能性がある）
    messages.append(response_message)

    # ステップ3:モデルがツールを使いたいかどうかを確認
    if response_message.tool_calls:
        # ステップ4:ツール呼び出しを抽出し、評価を行う
        for tool_call in response_message.tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            function_response = function_to_call(**function_args)

            # ステップ5:モデルが今後のターンで関数のレスポンスを
            # 確認できるように、関数のレスポンスで会話を下空調
            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": function_response,
            })

    return messages

def run_conversation(client):
    # メッセージを初期化し、エージェントの機能を説明する序文を作成
    messages = [
        {
            "role": "system",
            "content": "あなたは、役に立つサーモスタットアシスタントです",
        }
    ]
    while True:
        # ユーザー入力を要求し、メッセージに追加
        user_input = input(">> ")
        if user_input == "":
            break
        messages.append({
            "role": "user",
            "content": user_input,
        })
        while True:
            new_messages = process_messages(client, messages)

            last_message = new_messages[-1]
            if not isinstance(last_message, ChatCompletionMessage):
                continue # これは、単なるツールレスポンスメッセージ
            if last_message.content is not None:
                print(last_message.content)
                #　ツール呼び出しでない場合、次のメッセージを待機
                # ブレークし、入力を待機
                if last_message.tool_calls is None:
                    break

    return messages

def main():
    run_conversation(client)

if __name__ == "__main__":
    main()
