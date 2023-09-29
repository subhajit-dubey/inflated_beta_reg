# inflated_beta_reg
Imitating the PROC NLMIXED output from SAS required for Inflated Beta Regression Modeling 
using Python.

For detailed documentation on module usage and user guide follow the [link](https://github.com/pages/subhajit-dubey/inflated_beta_reg)

### SAS Code

<code>

    proc nlmixed data=work.import
        tech=quanew maxiter=3000 maxfunc=3000 qtol=0.0001
        itdetails
        ;
        parms 	B0 - B4 = 0.0001
                pie = 0.2
                kesai = 0.3
                phi = 2
            ;
        cov_mu = B0 + B1 * Col1 + B2 * Col2 
                    + B3 * Col3 + B4 * Col4;
        mu = logistic(cov_mu);
        
        if ColY = 0 then loglikefun = log(pie)+log(1-kesai);
        if ColY >= 1 then loglikefun = log(pie)+log(kesai);
        if 0 < ColY < 1 then loglikefun = log(1-pie) + lgamma(phi) -
            lgamma(mu*phi) - lgamma((1-mu)*phi) +
            (mu*phi-1)*log(ColY) + ((1-mu)*phi-1)*log(1-ColY);
            
        model ColY ~ general(loglikefun);
        ods output IterHistory = iter_history;

    run;
    quit;
</code>

### SAS Output:

![img1](https://github.com/subhajit-dubey/inflated_beta_reg/blob/master/img/img1.jpg)

### Python Output

![img2](https://github.com/subhajit-dubey/inflated_beta_reg/blob/main/img/img2.jpg)