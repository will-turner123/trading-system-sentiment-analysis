from datetime import datetime, timedelta

def yesterday():
    today = datetime.now()
    yesterday = today - timedelta(1).split(' ')
    return yesterday[0]

print(yesterday())