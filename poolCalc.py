import sys

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print('Arguments: img_dim, filetr_sz, stride')
        exit()

    ans = int(sys.argv[1])
    ans = ans - int(sys.argv[2])
    ans = ans / int(sys.argv[3])
    ans = ans + 1

    print('output dim maxpool: ' + str(ans))

