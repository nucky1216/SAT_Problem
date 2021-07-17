import numpy as np
from pysat.solvers import *
import time
from pysat.formula import CNF


VARIBLES=3

def RandomSign(num,NumListSpecific=None):
    SignsList=[-1,1]
    if NumListSpecific is None:
        NumList = range(1, num + 1)
        VariablesNumList=[1,2,3]
        VariablesNum=np.random.choice(VariablesNumList)

    else:
        NumList=NumListSpecific
        VariablesNum=len(NumList)

    sign = np.random.choice(SignsList, VariablesNum, replace=True)
    x = np.random.choice(NumList, VariablesNum, replace=False)

    result=x*sign
    out=[]
    for r in result:
        out.append(int(r))

    return out,x
def Produce3NF(num_arrays,num_cluases,file=None):


    text='p cnf '+str(num_arrays)+' '+str(num_cluases)
    text+='\n\n'

    g=Glucose3(use_timer=True)
    CheckVaribleNum=[]
    count_clause=0
    for i in range(num_cluases):
        strs=''

        clause,abs_clause= RandomSign(num_arrays)
        CheckVaribleNum.extend(abs_clause)
        g.add_clause(clause)
        count_clause+=1

        strs=str(clause)+' 0\n'
        text+=strs
        CheckVaribleNum=list(set(CheckVaribleNum))

    if len(CheckVaribleNum) < VARIBLES:
        ListTotal=range(1,VARIBLES)
        RemainList=list(set(ListTotal)-set(CheckVaribleNum))

        ClauseElems = []
        for idx,remain in enumerate(RemainList):
            ClauseElems.append(remain)
            if len(ClauseElems)==3 or idx==len(RemainList)-1:
                re_clause,abs_reClause=RandomSign(-1,NumListSpecific=ClauseElems)
                g.add_clause(re_clause)
                num_cluases += 1

                ClauseElems=[]


                strs = str(re_clause) + ' 0\n'
                text += strs
                #CheckVaribleNum = list(set(CheckVaribleNum))



    print(text)
    print('========================SAT=====================')
    start_time=time.time()
    Satisfy=g.solve()

    print('Satisify:',Satisfy)
    print('Variables:',VARIBLES,'Num_clause:',num_cluases)
    print('accu time:',g.time_accum())
    print('call time:', g.call_time)

    print('Outer time:',time.time()-start_time)

    if Satisfy==False:
        print('Proof:',g.get_proof())
        print('Core:',g.get_core())
    else :
        print(g.get_model())
    print()
    if file!=None:
        with open(f'CNF\Phase_{file}.cnf','w') as f:
            #f.write(text)
            pass
def Test():
    f=CNF()
    g=Glucose3(use_timer=True,with_proof=True)
    g.add_clause([-1,2,3])
    g.add_clause([1,3,4])
    g.add_clause([1, 3, -4])
    g.add_clause([1, -3, 4])

    g.add_clause([1, -3, -4])
    g.add_clause([-2, -3, 4])
    g.add_clause([-1, 2, -3])
    g.add_clause([-1,-2,3])

    start_time = time.time()
    Satisfy = g.solve(assumptions=[-1,-3])

    print('Satisify:', Satisfy)

    print('accu time:', g.time_accum())
    print('call time:', g.call_time)

    print('Outer time:', time.time() - start_time)


    print('Proof:', g.get_proof())
    print('Core:', g.get_core())
    print('staus:',g.get_status())

    print(g.get_model())

if __name__=='__main__':
    print('Hello')
    #
    # print(RandomSign(4))
    #
    Num_Formulas=10000
    # for i in range(Num_Formulas):
    #     Produce3NF(VARIBLES, 50+i,file=i)

    #Produce3NF(VARIBLES, 4)
    Test()

