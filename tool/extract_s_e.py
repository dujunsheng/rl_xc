# 提取route的起点和终点

result = set()
with open('all_route.csv', 'r')as f:
    lines = f.readlines()
    for line in lines:
        result.add((line.split(' ')[0].strip(), line.split(' ')[-1].strip()))
print(result)
