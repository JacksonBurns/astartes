import codecs

with codecs.open("temp.txt", encoding='utf-8') as file:
    lines = file.readlines()

coverage = lines[-1].split()[-1].replace("%", "")

with open("temp2.txt", 'w') as file:
    file.write(coverage)
