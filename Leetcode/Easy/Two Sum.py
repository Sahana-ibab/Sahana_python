class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        seen={}
        for inx, num in enumerate(nums):
            n = target - num
            if n in seen:
                return [seen[n],inx]
            seen[num]= inx


def main():
    sol= Solution()
    result = sol.twoSum([1,2,3], 5)
    print(result)


if __name__ == '__main__':
    main()



import numpy as np

def input_func():
    n = int(input("Enter the no of inputs: "))
    x = []
    for i in range(n):
        x.append(float(input(f"Enter x{i+1}: ")))
    print("Entered inputs: ", x)

    l = int(input("Entered the number of Hidden layers: "))
    B=[]
    W=[]
    for j in range(l):
        u = int(input(f"Enter the number of neurons in layer-{j+1}: "))
        w = np.zeros((u,n))
        b = np.zeros(u)
        for k in range(u):
            for i in range(n):
                w[k][i]=float(input(f"Enter w{k+1}{i+1} of Layer{j+1}: "))
            b[k]=(int(input(f"Enter the bias term for {k+1} neuron of {j+1} layer ")))
        B.append(list(b))
        n=u
        W.append(list(w))

    O=int(input("Enter the number of neurons in the Output layer: "))
    ow=[]
    bo=[]
    for o in range(O):
        p=[]
        for h in range(n):
            p.append(float(input(f"Enter weight of last layer: w{o+1} ")))
            bo.append(int(input(f"Enter the bias term for {o+1} neuron of {h+1} layer ")))

        ow.append(p)

    return x, W, ow, B, bo

def matrix_multi(x, w):
    return np.dot(x, w)

def relu(z):
    return np.maximum(0,z)

def softmax_func(z):
    return np.exp(z)/(np.sum(np.exp(z)))

def feed_forward(inputs):
    x, W, ow , B, bo= inputs
    Z=[]
    x=np.array(x)

    for i in range(len(W)):
        z=[]
        for j in range(len(W[i])):
            m = np.sum(matrix_multi(x, W[i][j]),B)
            z.append(m)
        r=relu(z)
        x=np.array(r)
        Z.append(r)

    S=[]
    for l in range(len(ow)):
        m = matrix_multi(x, ow[l])+bo
        S.append(m)
    soft=softmax_func(S)
    return soft

def main():
    inputs=input_func()
    print("-----feed forward process-----\n")
    print("Default activation function: Relu for all the hidden layers and softmax for the O/P layer.\n")
    ffp = feed_forward(inputs)
    print("Result: ", ffp)

if __name__ == '__main__':
    main()