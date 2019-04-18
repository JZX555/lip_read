# encoding=utf8
def percent(current, total):

    if current == total - 1:
        current = total
    bar_length = 20
    hashes = '#' * int(current / total * bar_length)
    spaces = ' ' * (bar_length - len(hashes))
    print(" -----------------------------------------------------------------")
    print(
        "\rPercent: [%s] %d%%" % (hashes + spaces, int(100 * current / total)))
    print(" -----------------------------------------------------------------")
