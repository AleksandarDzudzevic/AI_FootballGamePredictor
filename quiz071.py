from random import randint
import time
class ipv6machine():
    def __init__(self,n:int):
        self.n=n
        self.ip_list=[]
    def get_addresses(self):
        dict={1:'1',2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',10:"A",11:"B",12:"C",13:"D",14:"E",15:"F"}
        cnt=0
        address=""
        while len(self.ip_list)!=self.n:
            for digits in range(0,32):
                address+=f"{dict[randint(1,15)]}"
                cnt+=1
                if cnt==4:
                    address+=":"
                    cnt=0
            if address not in self.ip_list:
                self.ip_list.append(address)
            address=""

        return self.ip_list
start_time = time.time()
print(ipv6machine(n=10000).get_addresses())
end_time = time.time()

execution_time = end_time - start_time
print("Execution time:", execution_time, "seconds")

