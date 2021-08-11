from itertools import islice
from decimal import Decimal
import math
f = open("price_information.txt")
list = []
v = 0
for value in islice(f, 0, 6):
    Process_data = value.replace('\n', '')
    list.append(Process_data)
O_price = Decimal(list[0])
C_price = Decimal(list[1])
CNN_accuary = Decimal(list[2])
GRU_accuary = Decimal(list[3])
LSTM_accuary = Decimal(list[4])
accuary_max = max(CNN_accuary, GRU_accuary, LSTM_accuary)

# Three parameters
V_value = Decimal('%.3f' % (O_price/accuary_max))
s = O_price-C_price
M = 32*(V_value-O_price)/(3*(4*V_value-1))
# Formula
a1 = (3*V_value+s)/4
a2 = 9*pow(V_value-s, 2)
a3 = 3*pow(1-s, 2)*(3*V_value-2-s)*M
c = -(1/12)*math.sqrt(a3+a2)
final_price = '%.3f'% (a1+int(c))
print('Recommended price: ' + final_price + ' Pounds')
