#!usr/bin/env python3

# This small script is set of generators examples in python
import itertools as it

print(' ')

a = range(0,20)
# a is range, a lazy generator in python3.x, meaning that 
# we can start to use the value before the generation of all of them
print('a : {}'.format(a))
for i in a :
    print('  {}'.format(i))
print(' ')

a = range(0,20)
# b is a generator
b = (x**2 for x in a)
type(b)
print('b : {}'.format(b))
for i in b:
    print('  {}'.format(i))
print(' ')

a = range(0,20)
# c is a list
c = [x**2 for x in a]
print('c : {}'.format(c))
for i in c:
    print('  {}'.format(i))
print(' ')

a = range(0,20)
b = (x**2 for x in a) 
d = enumerate(b)
print('d : {}'.format(d))
for i in d:
    print('  {}'.format(i))
print(' ')

# d is enumarate. The enumerate object yields pairs containing a count (from
# start, which defaults to zero) and a value yielded by the iterable argument.
d = enumerate((x**2 for x in range(0,20)))
print('d : {}'.format(d))
for (idx,val) in it.islice(d,10,15):
    print('{}^2 = {}'.format(idx,val))
print(' ')

l = [1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
primes = (nb for nb in l)
print('primes : {}'.format(primes))
try:
    while True:
        current = next(primes)
        print('  {}'.format(current))
except StopIteration:
    print('Done with prime numbers')
print(' ')

a = range(0,20)
b = (x**2 for x in range(0,20)) 
# d is zip here, a set of tuples
d = zip(a,b)
print('d : {}'.format(d))
for i in d:
    print('  {}'.format(i))
print(' ')
