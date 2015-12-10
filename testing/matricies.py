
import numpy as np

if __name__ == "__main__":
    
    batch_size=32
    x = np.random.randn(batch_size,8)
    # a = np.zeros((32,1),dtype=np.dtype('int32'))
    a = np.random.randint(0,7,size=(batch_size,1))
    
    print "A: " + str(a)
    print "X: " + str(x)
    print x[a]
    print a.shape[0]
    print x[np.arange(x.shape[0]), a.reshape((-1,))].reshape((-1, 1))