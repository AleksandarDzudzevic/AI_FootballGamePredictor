def ip4vmachine():
    address=[]
    for ip in range(256**4):
        a=ip%256
        b=ip//256%256
        c=ip//256//256%256
        d=ip//256//256//256%256
        address.append(f"{d}.{c}.{b}.{a}")
    return(address)
print(ip4vmachine())
# IT WILL NEVER LOAD BECAUSE it is over 4 billion addresses!!!!


