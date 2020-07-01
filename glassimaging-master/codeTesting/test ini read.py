try:
    with open('platform.ini', 'r') as f:
        platform = f.readline()
except:
    platform = 'unknown'

print(platform)
