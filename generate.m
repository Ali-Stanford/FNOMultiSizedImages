n=56;

n_data = 5000;

counter = 0;

while counter < n_data + 1


    a3 = rand(n,n,n);
    thr = 0.45;
    lc =  5; 
    sigma = 2.0 ;      
    result3d = imgaussfilt3(a3,sigma,'FilterSize',lc);	
    result3d = (result3d - min(min(min(result3d))))/(max(max(max(result3d)))-min(min(min(result3d))));
	result3d = imbinarize(result3d,thr);
	porosity3d = sum(result3d==0,'all')/n/n/n;
      
      if porosity3d > 0.20
          continue;
      end

      if  porosity3d < 0.124
          continue ;
      end
      
      counter = counter + 1 ;

	id = fopen('3Dporous'+string(counter)+'.dat','w');
	for i = 1:n
		for j = 1:n
			for k = 1:n 
				fprintf(id,'%i\n',result3d(i,j,k));
			end
		end
	end
	
     fclose(id);
end