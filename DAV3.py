# files needed:  A file named "y.txt" that has the vector to be coded saved as one element per row of the text file--
# (i.e, each element separated by a new line character)
# and another file named "A.txt" that has the elements of A stored as: row elements separated by a comma, each row on a new line
# A contains the basis vectors as its columns

import numpy as np

def feature_sign_search(x,theta,y,A):
    #x are the weights of the basis vectors
    #theta are the signs of the x
    #step 1:initialization
    gamma=10**-8
    itr=0
    active=[]
    while (True):
    #step 2: from zero coefficients of x select i=argmax (.. )
        active=list(active)
        theta=np.sign(x)
        diff=np.zeros(len(x))
        for ind in range(0,len(x)):
            sum_overj=0
            for j in range(0,len(y)):
                sum_overi=0
                for i in range(0,len(x)):
                    sum_overi+=A[j,i]*x[i]
                sum_overj+=(y[j]-sum_overi)*(-A[j,ind])
            diff[ind]=2*sum_overj
    
        diff_mod=abs(diff)
        index=np.argmax(diff_mod)    #selecting i as the argmax
        if diff[index]>gamma:
            theta[index]=-1
            active.append(index)
        elif diff[index]<-gamma:
            theta[index]=1
            active.append(index)
        active=np.asarray(active)  
        
        while (True):    
        #step 3: feature-sign step
            itr+=1
            A_cap=A[:,active]
            x_cap=x[active]
            theta_cap=np.sign(x_cap)
            curr_sign=theta_cap
#            print("A_cap: ",A_cap)
#            print("x_cap: ",x_cap)
#            print("theta_cap: ",theta_cap)
            
            term1=np.linalg.inv(np.matmul(A_cap.T,A_cap))
            term2=np.matmul(A_cap.T,y)-(gamma/2)*theta_cap
#            print("term1: ",term1, "term2: ",term2)
            xnew_cap=np.matmul(term1,term2)
#            print("x_cap= ",x_cap)
#            print("xnew_cap=",xnew_cap)
        
            direction=xnew_cap-x_cap
            min_obj=np.linalg.norm(y-np.matmul(A_cap,x_cap))**2 + gamma*np.dot(theta_cap,x_cap)
            min_x=x_cap
#            print("\nIteration: ",itr)
#            print("active indices: ", active)
        
            for frac in np.arange(0,1.1,0.1):
                xtry=x_cap+frac*direction
                theta_cap=np.sign(xtry)
#                print("xtry= ",xtry)
                if min(np.sign(xtry)==np.sign(x_cap))==False:
                    obj=np.linalg.norm(y-np.matmul(A_cap,xtry))**2 + gamma*np.dot(theta_cap,xtry)
#                    print("obj: ",obj)
                    if obj<min_obj:
                        min_x=xtry
                        min_obj=obj
            x_cap=min_x
            theta_cap=np.sign(x_cap)
            new_sign=theta_cap
#            print("x_cap optimized: ",min_x)
            j=0
            for i in active:
                x[i]=x_cap[j]
                j=j+1
            activenew=[]
            theta=np.sign(x)
        
            for i in active:
                if x[i]!=0:
                    activenew.append(i)
#                else:
#                    print("\nShould not be coming here i guess\n")
            activenew=np.asarray(activenew)
            active=activenew
            nonzero_x=x[x!=0]
            nonz_diff=np.zeros(len(nonzero_x))
            cond1=True
        
            for ind in range(0,len(nonzero_x)):
                sum_overj=0
                for j in range(0,len(y)):
                    sum_overi=0
                    for i in range(0,len(x)):
                        sum_overi+=A[j,i]*x[i]
                    sum_overj+=(y[j]-sum_overi)*(-A[j,ind])
                nonz_diff[ind]=2*sum_overj
                if nonz_diff[ind] + gamma*np.sign(nonzero_x[ind]) !=0:
#                if nonz_diff[ind] + gamma*np.sign(nonzero_x[ind])>0.01 or nonz_diff[ind] + gamma*np.sign(nonzero_x[ind])<-0.01:
                    cond1=False
#            print("nonz_diff=",nonz_diff)
#            print("sign of x: ",np.sign(nonzero_x))
            if cond1 == True or min(curr_sign==new_sign)==True:
#                print("cond1 satisfied!! breaking out")
#                print(cond1)
                break
    
#        print("broke out successfully")
    
        zero_x=x[x==0]
        z_diff=np.zeros(len(zero_x))
        cond2=True
        
        for ind in range(0,len(zero_x)):
            sum_overj=0
            for j in range(0,len(y)):
                sum_overi=0
                for i in range(0,len(x)):
                    sum_overi+=A[j,i]*x[i]
                sum_overj+=(y[j]-sum_overi)*(-A[j,ind])
            z_diff[ind]=2*sum_overj
            if abs(z_diff[ind]) > gamma:
                cond2 = False
        
        if cond2 == True:# or itr>50:
#            print("cond2 satisfied!! breaking out!!!")
#            print(cond2)
            break
#    print("broke out successfully")
    return x
    
    
A = np.loadtxt('A.txt',delimiter=',')
yd,xd = A.shape
y = np.loadtxt('y.txt',delimiter=',')
x = np.zeros(xd)
theta = np.zeros(xd)
#yd should be equal to the dimensions of y
#A contains the basis vectors as its column vectors
x_sol = feature_sign_search(x,theta,y,A)

print("x: ",x_sol)
print("A: ",A)
print("y: ",y)
print("A.x: ",np.matmul(A,x_sol))
