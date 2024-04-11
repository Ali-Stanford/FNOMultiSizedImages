import operator

### Pam.py ###
data_number = 1250
nx = 40 #48, 56
ny = 40 #48, 56
nz = 40 #48, 56
dx = 0.003

for i in range(data_number):   
    with open(str(1+i)+'.pam', 'w') as f:
        f.write('3Dporous'+str(1+i)+'.dat'+'\n')
        f.write('0\n')
        f.write(str(nx)+' '+str(ny)+' '+str(nz)+'\n')
        f.write(str(dx)+'\n')
        f.write('1\n')
        f.write('15')
    f.close()       
     
### Script.py ###
for i in range(data_number):
    
    with open('simulation'+str(i+1)+'.sh', 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('#SBATCH -p serc\n')
        f.write('#SBATCH -c 1\n')
        f.write('#SBATCH --mem=16GB\n')
        f.write('#SBATCH --time=0:10:00\n')
        f.write('#SBATCH --output=ai-%j.out\n')
        f.write('./perm_SP_IFP '+str(i+1)+'.pam\n')

    f.close()

### Job.py ###
with open('runData.sh', 'w') as f:
    
    f.write('#!/bin/bash\n')

    for i in range(data_number):   
    
        f.write('sbatch simulation'+str(i+1)+'.sh\n')

f.close()  
