function [ nmi ] = calculateNMI(cluster,true_cluster)
n = size(cluster,1);
K1 = size(unique(cluster),1);
K2 = size(unique(true_cluster),1);

I = 0;
for i=1:K1    %cluster
    for j=1:K2    %label
        wkcj = 0;
		wk = 0;
		cj = 0;
		N = 0;
        for p=1:n
            N = N + 1;
            if (cluster(p) == i) && (true_cluster(p) == j)
                wkcj = wkcj + 1;
            end;
            if cluster(p) == i
                wk = wk + 1;
            end;
            if true_cluster(p) == j
                cj = cj + 1;
            end;
        end;
        if wkcj > 0
			I = I + (wkcj/N)*log((N*wkcj)/(wk*cj) + eps);
        end;
    end;
end;

H1 = 0;
for i=1:K1
    wk = 0;
    N = 0;
    for p=1:n
        N = N + 1;
        if cluster(p) == i
            wk = wk + 1;
        end;
    end;
    H1 = H1 + (wk/N)*log(wk/N + eps);
end;
H1 = H1 * (-1);

H2 = 0;
for j=1:K2
    cj = 0;
    N = 0;
    for p=1:n
        N = N + 1;
        if true_cluster(p) == j
            cj = cj + 1;
        end;
    end;
    H2 = H2 + (cj/N) * log(cj/N + eps);
end;
H2 = H2 * (-1);
nmi = 2 * I / (H1 + H2);
end

