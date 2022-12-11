function [R_est,t_est]=Direct_ICP(p1,p2,imu_R,imu_t,weight) 
 
% p1为上一帧的点云信息，p1为cell变量，包含cluster个点团信息，第i个点团维度是3×n_i，n_i为点团中点的数量 
% p2为下一帧的点云信息，同理p2为cell变量，包含cluster个点团信息，第i个点团维度是3×l_i，l_i为点团中点的数量 
% imu为IMU测得的相对位姿 
% weight为给imu信息加的权重 
 
m=10; 
cluster=length(p1); 
b=zeros(m*cluster,1);  % m为给每个点团选的虚拟点的数量 
A=zeros(m*cluster,16); 
x=zeros(3,m); 
for i=1:cluster 
    points1=cell2mat(p1(i)); 
    points2=cell2mat(p2(i)); 
    n_i=length(points1); 
    l_i=length(points2); 
    bar_p2=sum(points2,2)/l_i; 
    xmin=min(points1(1,:)); 
    xmax=max(points1(1,:)); 
    ymin=min(points1(2,:)); 
    ymax=max(points1(2,:)); 
    zmin=min(points1(3,:)); 
    zmax=max(points1(3,:)); 
    for j=1:m 
        x(1,j)=(xmax-xmin)*rand+xmin; 
        x(2,j)=(ymax-ymin)*rand+ymin; 
        x(3,j)=(zmax-zmin)*rand+zmin; 
        for k=1:n_i 
            b((i-1)*m+j)=b((i-1)*m+j)+norm(x(:,j)-points1(:,k))^2/n_i; 
        end 
        for k=1:l_i 
            b((i-1)*m+j)=b((i-1)*m+j)-norm(points2(:,k))^2/l_i; 
        end 
        b((i-1)*m+j)=b((i-1)*m+j)-norm(x(:,j))^2; 
        L=kron(x(:,j)',eye(3)); 
        A((i-1)*m+j,:)=[-2*bar_p2'*L -2*bar_p2' 2*x(:,j)' 1]; 
    end 
end 
 
theta_imu=[vec(imu_R)' imu_t' imu_t'*imu_R norm(imu_t)^2]'; 
A_bar=[A;weight*eye(16)]; 
b_bar=[b;weight*theta_imu]; 
estimate=(A_bar'*A_bar)\A_bar'*b_bar; 
R_est=[estimate(1:3) estimate(4:6) estimate(7:9)]; 
t_est=estimate(10:12);