with open("temp.txt") as file:
    lines = file.readlines()

coverage = lines[-1].split()[-1]

print('Test coverage is', coverage, 'percent.')

with open("temp2.txt", 'w') as file:
    file.write(coverage)
