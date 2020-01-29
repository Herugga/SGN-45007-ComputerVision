function l = linefitlsq(x,y)

xm=mean(x(:));
ym=mean(y(:));

U=[x(:)-xm y(:)-ym];

[V,D]=eig(U'*U);
[minv,minid]=min(diag(D));
ev=V(:,minid);
a=ev(1);b=ev(2);

d=a*xm+b*ym;

l=[a;b;-d];

end

