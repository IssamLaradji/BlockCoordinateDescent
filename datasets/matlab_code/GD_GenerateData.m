function [x,y]=GD_GenerateData(type,num,dim,ClassProb,variance)
% generates data from different densities with different number of classes
% can either be called as
% Input: type ,     corresponding number of density
%        num  ,     number of data points
%        dim  ,     Dimension of the ambient space (and dimension of the noise)  
%        ClassProb, class probabilities
%        variance , variance of the noise
%
%  [x,y]=GenerateData(type,num,dim) : In this case class probabilities are
%                                 balanced
%  [x,y]=GenerateData(type,num,dim,ClassProb) In this case data is generated
%                                         randomly according to class probabilities
%
%  [x,y]=GenerateData(type,num,dim,ClassProb,variance) In this case data is generated
%                                         randomly according to class
%                                         probabilities and the Gaussian
%                                         noise is specified by variance
%
% Density: 1   Two Lines opposite to each other
%          2   The two moons data with Gaussian noise
%          3   Three Gaussians of the same variance (two Gaussians can
%              be generated using ClassProb=[0.5, 0.5, 0])
%          4   Three Gaussians of different variance (two Gaussians can be
%              generated using ClassProb=[0.5, 0.5, 0]
%

type=floor(type);

% check arguments
DEFAULT_CLASS=0;
switch(nargin)
  case 0, density=1; num=100; dim=3;  DEFAULT_CLASS=1; variance=0.04;
  case 1, num=100; dim=3;  DEFAULT_CLASS=1; variance=0.04;
  case 2, dim =3;  DEFAULT_CLASS=1; variance=0.04;
  case 3, DEFAULT_CLASS=1; variance=0.04;
  case 4, DEFAULT_CLASS=0; variance=0.04;
end

% check class probabilities
if(DEFAULT_CLASS==0)
  num_classes=length(ClassProb);
  if(sum(ClassProb)~=1)
   display('Class Probabilities do not sum to 1 !');
   if(sum(ClassProb)~=0)
    ClassProb=ClassProb/sum(ClassProb);
   else
    ClassProb=ones(num_classes,1)/num_classes;
   end
  end
end

switch(type)
  case 1, if(DEFAULT_CLASS==1 | num_classes~=2)
           ClassProb=[0.5,0.5]; num_classes=2;
          end
  case 2, if(DEFAULT_CLASS==1 | num_classes~=2)
           ClassProb=[0.5,0.5]; num_classes=2;
          end
  case 3, if(DEFAULT_CLASS==1 | num_classes<2)
           ClassProb=[0.33,0.33,0.34]; num_classes=3;
          end
end

% generate random numbers to determine the number of points according to
% the class probabilities
num = floor(num);  % convert to integer just in case num is not integer
pos = rand(num,1); % generate num points in the unit interval 

numPtsClasses=zeros(1,num_classes);

for i=1:num_classes
 if(i==1)
  numPtsClasses(i)=sum(pos >= 0 & pos<ClassProb(i));
 else
  numPtsClasses(i)=sum(pos >= sum(ClassProb(1:i-1)) & pos<sum(ClassProb(1:i)));  
 end
end
  


 
% Generate now the points

y=zeros(num,1);
switch(type)       
    % two lines 
    case 1, % class 1, lower line + Gaussian noise
            x=[rand(1,numPtsClasses(1));   zeros(1,numPtsClasses(1)); zeros(dim-2,numPtsClasses(1))]+ sqrt(variance)*randn(dim,numPtsClasses(1));  
            y(1:numPtsClasses(1),1)=1;
            % class 2, upper line + Gaussian noise
            x(:,numPtsClasses(1)+1:num)=[rand(1,numPtsClasses(2)); 0.5*ones(1,numPtsClasses(2)); zeros(dim-2,numPtsClasses(2))]+sqrt(variance)*randn(dim,numPtsClasses(2));
            y(numPtsClasses(1)+1:num,1)=2;
            display(['Generation of two one dimensional lines in ',num2str(dim),' dimensions with Gaussian noise of variance',num2str(variance)]);
            display(['Number of points in Class 1: ',num2str(numPtsClasses(1)),'; Class 2: ',num2str(numPtsClasses(2))]);
            
   %generate two-moon problem
   case 2, num_pos=numPtsClasses(1); num_neg=numPtsClasses(2);
           %radii=0.75+0.15*randn(num_pos+num_neg,1);
           radii=ones(num,1);%+0.22*randn(num_pos+num_neg,1);
           phi  =rand(num_pos+num_neg,1).*pi;
           
           % the following generates two half circles in two dimensions in two-moon form
           x=zeros(dim,num);
           for i=1:num_pos
             x(1,i)=radii(i)*cos(phi(i));
             x(2,i)=radii(i)*sin(phi(i));
             y(i,1)=1;
           end
           for i=num_pos+1:num_neg+num_pos
             x(1,i)=1+radii(i)*cos(phi(i));
             x(2,i)=-radii(i)*sin(phi(i))+0.5;
             y(i,1)=2;
           end
           % add Gaussian noise
           x=x + sqrt(variance)*randn(dim,num);
           display(['Generation of two moons in ',num2str(dim),' dimensions with Gaussian noise of variance',num2str(variance)]);
           display(['Number of points in Class 1: ',num2str(numPtsClasses(1)),'; Class 2: ',num2str(numPtsClasses(2))]);
          
    % three points with disturbed by isotropic Gaussian noise
    case 3, a=numPtsClasses(1); b=numPtsClasses(2); c=numPtsClasses(3); 
            %C1=randn(dim,dim).^2; C1=(C1+C1')/(3*sqrt(dim));
            %C2=randn(dim,dim).^2; C2=(C2+C2')/(3*sqrt(dim));
            %C3=randn(dim,dim).^2; C3=(C3+C3')/(3*sqrt(dim));
            x(:,1:a)      =0.6*randn(dim,a) - [1.1*ones(2,a); zeros(dim-2,a)]; y(1:a,1)=1;
            x(:,a+1:a+b)  =0.6*randn(dim,b) + [ones(2,b); zeros(dim-2,b)]; y(a+1:a+b,1)=2;
            x(:,a+b+1:num)=0.6*randn(dim,c) + [2*ones(1,c); -2*ones(1,c); zeros(dim-2,c)]; y(a+b+1:num,1)=3;
            display(['Generation of three isotropic Gaussians of equal variance 0.36 in ',num2str(dim),' dimensions']);
            display(['Number of points in Class 1: ',num2str(a),'; Class 2: ',num2str(b),'; Class 3: ',num2str(c)]);
            
    % three isotropic Gaussians with different variances
    case 4, a=numPtsClasses(1); b=numPtsClasses(2); c=numPtsClasses(3); 
            %C1=randn(dim,dim).^2; C1=(C1+C1')/(3*sqrt(dim));
            %C2=randn(dim,dim).^2; C2=(C2+C2')/(3*sqrt(dim));
            %C3=randn(dim,dim).^2; C3=(C3+C3')/(3*sqrt(dim));
            x(:,1:a)      =0.6*randn(dim,a) - [1.1*ones(2,a); zeros(dim-2,a)]; y(1:a,1)=1;
            x(:,a+1:a+b)  =0.4*randn(dim,b) + [ones(2,b); zeros(dim-2,b)]; y(a+1:a+b,1)=2;
            x(:,a+b+1:num)=0.2*randn(dim,c) + [2*ones(1,c); -2*ones(1,c); zeros(dim-2,c)]; y(a+b+1:num,1)=3;
            display(['Generation of three isotropic Gaussians with variances 0.36, 0.16 and 0.04 in ',num2str(dim),' dimensions']);
            display(['Number of points in Class 1: ',num2str(a),'; Class 2: ',num2str(b),'; Class 3: ',num2str(c)]);
            
    % two points disturbed by isotropic Gaussian noise / classes go
    % through the middle of the Gaussian
    case 5, a=numPtsClasses(1); b=numPtsClasses(2); c=numPtsClasses(3); 
            %C1=randn(dim,dim).^2; C1=(C1+C1')/(3*sqrt(dim));
            %C2=randn(dim,dim).^2; C2=(C2+C2')/(3*sqrt(dim));
            %C3=randn(dim,dim).^2; C3=(C3+C3')/(3*sqrt(dim));
            
            x(:,1:a)      =0.6*randn(dim,a) - 1.5*[ones(1,a); zeros(dim-1,a)]; 
            x(:,a+1:a+b)  =0.6*randn(dim,b) + 1.5*[ones(1,b); zeros(dim-1,b)];
            for i=1:num
              if(x(2,i)<0) y(i)=1;
              else         y(i)=2;
              end
            end
            display(['Generation of three isotropic Gaussians of equal variance 0.36 in ',num2str(dim),' dimensions']);
            display(['Number of points in Class 1: ',num2str(a),'; Class 2: ',num2str(b),'; Class 3: ',num2str(c)]);
end
