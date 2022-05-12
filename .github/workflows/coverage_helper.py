import codecs

with codecs.open("temp.txt", encoding='utf-8') as file:
    lines = file.readlines()

print(lines)

coverage = lines[-1].split()[-1].replace("%", "")

print('Test coverage is', coverage, 'percent.')

with open("temp2.txt", 'w') as file:
    file.write(coverage)
