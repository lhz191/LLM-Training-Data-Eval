from googleapiclient import discovery


PERSPECTIVE_ATTRIBUTES = {
    "TOXICITY": {},              # 一般性毒性
    "SEVERE_TOXICITY": {},       # 严重毒性（threats, strong insults 等）
    "IDENTITY_ATTACK": {},       # 针对性别、种族、宗教等的攻击
    "INSULT": {},                # 辱骂、侮辱性语言
    "PROFANITY": {},             # 粗俗 / 脏话
    "THREAT": {},                # 威胁言语
    "SEXUALLY_EXPLICIT": {},     # 色情或露骨内容
    "FLIRTATION": {}             # 调情或性暗示
}



if __name__ == '__main__':
    client = discovery.build('commentanalyzer', 'v1alpha1', developerKey='AIzaSyDsAIT1hhGd9mdSaH3H4Mb9DyxamH9nLVM',
                             discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
                             static_discovery=False,)

    analyze_request = {
        'comment': {'text': "Some generated text here"},
        'requestedAttributes': PERSPECTIVE_ATTRIBUTES
    }

    response = client.comments().analyze(body=analyze_request).execute()

    # 输出每个维度分数
    for attr, score_info in response["attributeScores"].items():
        print(attr, score_info["summaryScore"]["value"])

    