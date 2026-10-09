"""Build the written submission from saved analysis tables (ReportLab only).

Run run_analysis.py first. Tables read the generated CSVs and JSON results.
"""
from pathlib import Path
import csv,json,html
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak,KeepTogether
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.utils import ImageReader

BASE=Path(__file__).resolve().parent
R=BASE/'results'
s=json.loads((R/'summary.json').read_text())
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodyA',fontName='Helvetica',fontSize=9.5,leading=12.5,spaceAfter=6))
styles.add(ParagraphStyle(name='SmallA',fontName='Helvetica',fontSize=8,leading=9.5,spaceAfter=4))
styles.add(ParagraphStyle(name='TitleA',fontName='Helvetica-Bold',fontSize=20,leading=24,spaceAfter=13,textColor=colors.HexColor('#12354a')))
styles['Heading2'].textColor=colors.HexColor('#12354a')
styles['Heading2'].spaceBefore=10
story=[]

def p(t,small=False):story.append(Paragraph(t,styles['SmallA' if small else 'BodyA']))
def title(t):story.append(Paragraph(t,styles['TitleA']))
def h(t):story.append(Paragraph(t,styles['Heading2']))
def page():story.append(PageBreak())
def fmt(v,d=4):
    try:
        x=float(v)
        if abs(x)>1e8:return f'{x:.3g}'
        return f'{x:,.{d}f}'
    except (ValueError,TypeError):return str(v)
def money(v):return f'${v:,.0f}' if v>=0 else f'-${-v:,.0f}'
def tbl(headers,rows,widths=None):
    data=[[Paragraph(html.escape(str(v)),styles['SmallA']) for v in row] for row in [headers]+rows]
    t=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e2ecf1')),('VALIGN',(0,0),(-1,-1),'TOP'),('BOTTOMPADDING',(0,0),(-1,-1),3),('TOPPADDING',(0,0),(-1,-1),3),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#829aa8')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f5f7f9')])]))
    story.extend([t,Spacer(1,9)])
def csvrows(name):
    with open(R/f'{name}.csv') as f:return list(csv.DictReader(f))
def mat(name):
    rows=csvrows(name);cols=list(rows[0]);tbl(['']+cols[1:],[[row[cols[0]]]+[fmt(row[c],5) for c in cols[1:]] for row in rows],[70]+[88]*(len(cols)-1))
def fig(name,width=490):
    path=BASE/'figures'/f'{name}.png';w,hg=ImageReader(str(path)).getSize();story.extend([Image(str(path),width=width,height=width*hg/w),Spacer(1,8)])

# Page 1
title('Assignment 2: Covariance, VaR, and Copulas')
p('FinTech 545 | Chuyu Qin',True)
p('All returns are arithmetic returns. Dollar risk is loss-positive: VaR = minus the lower P&amp;L quantile; ES = minus the mean of the worst alpha probability mass. A negative VaR denotes a gain at that quantile and is not set to zero. Unless labeled 1%, alpha is 5%. Simulations use 100,000 draws and fixed seed 5452026 (Problem 5 uses seed + 5).')
p('Sample variance uses n-1; skewness and excess kurtosis use empirical central moments. A normal distribution has skewness 0 and excess kurtosis 0.',True)
h('1. Correlations from mismatched histories - Predict (a-c)')
p('The joint-observation table below is computed before any correlation. Its diagonal is each series\'s available history. Only <b>81 days</b> contain all five series.')
rows=csvrows('p1_overlap');cols=list(rows[0]);tbl(['Series']+cols[1:],[[row[cols[0]]]+[row[c] for c in cols[1:]] for row in rows],[70]+[88]*5)
p('<b>(a)</b> Complete-case correlation is guaranteed positive semidefinite (PSD), provided each included series has positive variance. It is a standardized cross-product matrix built from one common set of rows: for every vector w, w\'Rw is a sum of squared projected observations and cannot be negative. It need not be strictly positive definite. Pairwise correlation has no such guarantee: each entry may use a different sample, so the entries need not correspond to any single multivariate data set.')
p('<b>(b)</b> If IDX exactly equaled the basket, one linear combination of the five returns would have zero variance, making the population correlation singular. Small tracking error makes its smallest eigenvalue small but positive. Thus the true matrix is close to the PSD boundary; comparatively small inconsistencies from the different observation windows can push an estimated eigenvalue below zero. This data set is especially exposed to that failure.')
p('<b>(c)</b> Correlations involving D should be least reliable because D has only 96 observations, with 83-96 observations per pair. C-D is the weakest by overlap count (83). A-B has 242 common observations. Counts are only a first reliability measure: the shorter D window can also represent different market conditions, and the missingness mechanism matters.')
h('Fit (d): how the matrices are constructed')
p('Complete-case correlation discards every row with any missing series. Pairwise correlation centers and scales each pair on its own overlapping rows. All reported eigenvalues are in ascending order. The following page gives both matrices and the ordinary, positive-definite Cholesky result.')

page();title('1. Correlations - Fit and reconcile')
h('(d) Estimated correlation matrices')
p('Complete case (81 common observations):',True);mat('p1_complete_correlation')
p('Pairwise (the overlaps shown on page 1):',True);mat('p1_pairwise_correlation')
for label,file in [('Complete case','complete_case'),('Pairwise','pairwise')]:
    vals=[float(r['eigenvalue']) for r in csvrows('p1_eigenvalues_'+file)]
    p(label+' eigenvalues: '+', '.join(f'{x:.7f}' for x in vals)+'.',True)
p('Ordinary Cholesky succeeds for complete case and fails for pairwise. The negative pairwise eigenvalue is far larger than numerical rounding error.')
h('(e-f) Tracking variance and PSD repairs')
p('Use each series\'s full available-history sample standard deviation, D = diag(s), and C = D R D. For w = (-0.40, -0.30, -0.20, -0.10, 1), tracking variance is w\'Cw. These are return-variance units, not dollars. All rows below use the same full-history standard deviations.')
tbl(['Correlation','Smallest eigenvalue','Frobenius change','Tracking variance'],[[r['matrix'],f"{r['min_eigenvalue']:.5g}",f"{r['frobenius_to_pair']:.7f}",f"{r['tracking_variance']:.7g}"] for r in s['p1']['summary']],[122,128,128,132])
p('Rebonato-Jackel truncates negative eigenvalues to zero and rescales to unit diagonal. Higham uses alternating PSD/unit-diagonal projections with Dykstra correction to minimize unweighted Frobenius distance. Its -2.1e-13 eigenvalue is numerical tolerance, effectively zero. Both repairs are PSD rather than necessarily positive definite; an ordinary Cholesky can still fail at the boundary. A PSD eigen-root is appropriate for simulation.',True)
p('<b>(g)</b> The pairwise tracking variance is -1.79937e-6. A variance cannot be negative; this is a concrete vector violating the PSD definition w\'Cw &gt;= 0. It diagnoses an internally inconsistent covariance estimate, not a negative economic risk. Repair changes that variance to about 6.8e-7.')

page();title('1. Reconciliation / 2. Volatility prediction')
h('1(h-i). What the repair changes')
changes=sorted(s['p1']['changes'],key=lambda r:abs(r['change']),reverse=True)
tbl(['Pair','Higham change','Joint days'],[[r['pair'],f"{r['change']:+.7f}",r['observations']] for r in changes],[170,170,170])
p('<b>(h)</b> A-IDX moves most (-0.007045), followed by C-IDX (-0.004765) and B-IDX (-0.004180). These are not the sparsest pairs: C-D moves only +0.001205. Higham is solving a geometric constraint problem using eigenvectors, eigenvalues, unit diagonal, and equal penalties on matrix entries. It was given no sample sizes or confidence intervals. Large loadings in the offending eigen-direction drive the corrections; they do not identify the statistically least trustworthy entries.')
p('<b>(i)</b> The complete-versus-pairwise Frobenius gap is 0.423935. The repairs move pairwise by only 0.016182 (Rebonato-Jackel) and 0.015520 (Higham), about 26-27 times less. The estimator choice matters much more in matrix distance for this sample. Higham is slightly closer than Rebonato-Jackel; neither establishes which missing-data estimator is statistically best.')
h('2. A volatility estimate after the regime changed - Predict (a-c)')
fig('p2_returns')
p('Prices give 500 returns; subtracting the full-sample mean of 0.000161876 leaves mean essentially zero. The plot suggests a regime change near day 460, so I use days 1-460 and the final 40 days as an approximate visual split, not a formally estimated change point.')
p('<b>(a) Predicted ascending VaR:</b> fitted t, historical, equally weighted normal, EW normal 0.97, EW normal 0.94. The first three are a tentative ordering. A t fit can emphasize the calm center and move additional mass beyond the 5% cutoff, while both full-history estimates largely reflect calm days. EW estimates put much more weight on the recent volatile period; lambda 0.94 reacts faster.')

page();title('2. Volatility - Fit and reconcile')
h('(b-c) Memory, weights, and moments')
tbl(['Lambda','Effective n','Half-life (days)','Weight: last 40'],[[r['lam'],f"{r['n_eff']:.3f}",f"{r['half_life']:.3f}",f"{100*r['recent40_weight']:.2f}%"] for r in s['p2']['ew']],[100,130,130,150])
p('For normalized weights w_i proportional to lambda^(500-i), n_eff = 1/sum(w_i^2), half-life = log(0.5)/log(lambda), and the last-40 weight is (1-lambda^40)/(1-lambda^500). The infinite-history approximation (1+lambda)/(1-lambda) is effectively identical here. EW covariance centers on the weighted mean and uses no Bessel correction, matching the course ewCovar; the normal VaR forecast mean is zero.')
m=s['p2']['moments'];tbl(['Mean','Sample variance','Skewness','Excess kurtosis'],[[f"{m['mean']:.2g}",f"{m['variance']:.9f}",f"{m['skewness']:.6f}",f"{m['excess_kurtosis']:.6f}"]],[127.5]*4)
p('The excess kurtosis of 2.132 is much larger than a typical single-regime normal sample would suggest (rough normal-reference standard error sqrt(24/500) = 0.219). The mild negative skew is less striking. One normal regime is not impossible in a finite sample, but the plot and kurtosis jointly argue strongly against that description.')
h('(d-e) Five dollar VaRs and the prediction check')
tbl(['Method','5% VaR'],[[r['method'],money(r['VaR'])] for r in s['p2']['var']],[330,180])
t=s['p2']['t'];p(f"The free-location Student t MLE gives nu = {t['nu']:.6f}, location = {t['mu']:.9f}, and scale = {t['scale']:.9f}. Its implied standard deviation is 0.01181774; scale is not standard deviation. Demeaning the data does not force the t MLE location to zero. Its VaR uses the fitted location.")
p('Actual ascending order is <b>t, equal normal, historical, EW 0.97, EW 0.94</b>. I missed the ordering of the middle two: historical VaR is $981 above the equal-normal estimate. The empirical 5% threshold reflects the realized downside observations, while the normal estimate smooths them into a symmetric shape. Excess kurtosis alone does not determine a distribution\'s 5% quantile. The t result illustrates that heavier far tails do not imply a larger VaR at every confidence level.')
h('(f-g) Regime mixture and estimation noise')
p('The within-regime standard deviations are 1.0178% over days 1-460 and 2.4289% over days 461-500. Pooling calm and volatile normals can create excess kurtosis: for a zero-mean variance mixture it is 3 E[sigma^4]/E[sigma^2]^2 - 3, which is positive when variance changes. The fitted t is a static summary of that pooled shape; it does not identify the regime or provide a responsive current-volatility forecast.')
p('The EW VaR gap is $2,244. Mapping sigma/sqrt(2 n_eff) into dollar VaR gives rough noise scales of $2,972 (0.97) and $4,515 (0.94), so the gap is smaller than either. These are individual-estimate approximations, not a formal test of their difference: the estimates use the same data and are correlated, and changing volatility weakens the stationary approximation.')
p('<b>Opinion:</b> I would use 0.94 for a trading limit because it responds faster to the recent jump, and 0.97 for a smoother capital number. The latter trades responsiveness for lower estimation noise; I would also review regime and stress assumptions rather than interpret the smaller estimate as sufficient capital by itself.')

page();title('3. When diversification raises VaR')
h('Predict (a-b): shape and default counts')
fig('p3_pnl')
p('Both P&amp;Ls have a large gain cluster and a much smaller default-loss region. Log counts make the rare outcomes visible. The CSV contains scenario-level variation around the stylized bond payoff; calculations use the supplied returns directly, without replacing them by binary payoffs.')
rows=csvrows('p3_moments');tbl(['Bond','Mean','Variance','Skewness','Excess kurtosis'],[[r[''],fmt(r['mean'],6),fmt(r['variance'],6),fmt(r['skewness'],4),fmt(r['excess_kurtosis'],4)] for r in rows],[62,112,112,112,112])
p('<b>(a)</b> Skewness around -4.8 to -4.9 and excess kurtosis above 22 are far from normal. A normal approximation spreads probability continuously around the mean, failing to represent the high-probability repayment cluster and low-probability default loss. Matching mean and variance does not recover that quantile structure.')
p('<b>(b)</b> A loses more than 20% in 406 scenarios (4.06%); B in 393 (3.93%). At least one loses more than 20% in 782 scenarios (7.82%), with 17 joint events (0.17%). The union exceeds 5%, although neither individual frequency does. I therefore predict higher 5% VaR after diversification. I predict lower ES for the diversified $2m choice because most default scenarios affect only half its capital; the counts support this intuition but do not determine loss severity by themselves.')
h('Fit (c-d): 5% historical risk and normal VaR')
tbl(['Position','Hist. VaR 5%','Hist. ES 5%','Normal VaR 5%'],[[r['position'],money(r['VaR5']),money(r['ES5']),money(r['normal_VaR5'])] for r in s['p3']['risk']],[132,126,126,126])
p('Historical VaR uses the 500th sorted P&amp;L out of 10,000; ES averages the worst 500. Normal VaR is -mean(P&amp;L) + 1.644854 sd(P&amp;L), using the sample standard deviation. No mean is removed in this problem. A negative historical VaR means the 5th-percentile outcome is still a positive gain; the sign convention is still loss-positive.')

page();title('3. Diversification - Reconcile')
h('(e) Historical risk at 1%')
tbl(['Position','VaR 1%','ES 1%'],[[r['position'],money(r['VaR1']),money(r['ES1'])] for r in s['p3']['risk']],[210,150,150])
p('At 1%, VaR uses the 100th sorted P&amp;L and ES averages the worst 100. Both concentrated and diversified quantiles now lie in the loss region.')
h('(f) Subadditivity at 5%')
a,b,con,div=s['p3']['risk']
p(f"For the two $1m positions, VaR(A) + VaR(B) = {money(a['VaR5']+b['VaR5'])}. The combined VaR is {money(div['VaR5'])}, which is larger by {money(div['VaR5']-a['VaR5']-b['VaR5'])}. Thus VaR(A+B) &lt;= VaR(A)+VaR(B) <b>fails</b>.")
p(f"ES(A) + ES(B) = {money(a['ES5']+b['ES5'])}, whereas ES(A+B) = {money(div['ES5'])}. The combined ES is lower by {money(a['ES5']+b['ES5']-div['ES5'])}; ES is <b>subadditive</b> here. The diversified $2m choice also has lower ES than the $2m A choice ({money(div['ES5'])} versus {money(con['ES5'])}), agreeing with the prediction.")
p('The mechanism is the cutoff: each standalone bond defaults in fewer than 5% of scenarios, so its 5% quantile sits in repayment outcomes and largely ignores default severity. Combining the bonds creates enough one-default days to put the portfolio\'s 5% quantile into the default region. ES instead integrates the whole bad tail, including the losses that VaR\'s standalone cutoff misses. Integrated-quantile ES handles the probability mass at a quantile boundary consistently and remains subadditive for these empirical scenarios.')
h('(g) What normal VaR gets right and wrong')
p('Normal VaR favors $1m in each: $143,466 versus $279,941 for $2m A. That agrees with the ES preference but disagrees with historical 5% VaR. The normal model recognizes a lower portfolio standard deviation, yet it is not right for the right distributional reason: it replaces default-event geometry with a symmetric bell curve. A sensible allocation preference does not validate its numerical tail estimate. In particular, its positive standalone 5% VaRs miss the empirical repayment-side cutoffs.')
h('(h) Why the result changes at 1%')
p(f"At 1%, combined VaR is {money(div['VaR1'])}, versus {money(a['VaR1']+b['VaR1'])} for the sum of the two standalone $1m VaRs. Subadditivity now holds in this sample. Relative to concentrating $2m in A, diversification lowers VaR from {money(con['VaR1'])} to {money(div['VaR1'])}.")
p('The 1% tail already contains defaults for each bond individually; spreading capital then reduces the size of a typical loss because simultaneous large defaults are rare (17 scenarios). VaR\'s subadditivity failure is not universal: it depends on where the probability cutoff falls relative to discrete or clustered loss events. Moving alpha can move the cutoff across such an event boundary. This example is therefore not evidence that diversification generally raises risk, nor that 1% VaR is always subadditive.')

page();title('4. Gaussian or t copula - Predict')
h('(a) Marginal moments and sensitivity')
rows=csvrows('p4_moments');tbl(['Asset','Mean','Variance','Skewness','Excess kurtosis'],[[r[''],fmt(r['mean'],7),fmt(r['variance'],8),fmt(r['skewness'],4),fmt(r['excess_kurtosis'],4)] for r in rows],[54,114,114,114,114])
p('X1 is nearly symmetric with slight negative excess kurtosis, suggesting a normal margin. X2 and X3 have positive excess kurtosis and suggest t margins. X2\'s skewness of -0.614 and kurtosis of 7.083 look particularly unusual relative to the other series, so I check a single-observation sensitivity before fitting.')
rows=csvrows('p4_outlier_sensitivity');tbl(['Asset','Removed data row','Removed return','Skew after removal','Excess after removal'],[[r['asset'],r['removed_row'],fmt(r['removed_return'],6),fmt(r['skewness'],4),fmt(r['excess_kurtosis'],4)] for r in rows],[54,100,114,121,121])
p('Data row 952 (excluding the CSV header) is each margin\'s largest absolute deviation from its median. Removing it changes X2 skewness to +0.056 and excess kurtosis to 1.710. Thus one observation explains much of X2\'s apparent asymmetry and extreme kurtosis, though remaining kurtosis still supports considering a t. This is a sensitivity diagnostic, not a reason to delete a valid observation; all fits retain all 1,000 rows.')
h('(b) Rank plots and joint-tail counts')
fig('p4_ranks')
tbl(['Pair','Both worst 2.5%','Both best 2.5%','Independence expected'],[[r['pair'],int(r['lower']),int(r['upper']),'0.625'] for r in s['p4']['tail'] if r['model']=='Observed ranks'],[120,130,130,130])
p('Ranks are scaled by n+1, so no coordinate equals 0 or 1 and each empirical 2.5% tail contains 25 observations. Independence implies 1000 x 0.025^2 = 0.625 joint-tail days. The corners are clustered, while the middle is more diffuse. Counts of 6-12 suggest connected extremes, so I predict the t copula will win. This is suggestive rather than proof: positive Gaussian dependence also produces more joint extremes than independence, and these are small tail counts.')

page();title('4. Copulas - Marginal and joint fits')
h('(c) Marginal MLE and AICc selection')
p('Normal location and scale are estimated by MLE (normal scale uses n, not n-1). The t additionally estimates degrees of freedom. Hence k = 2 for normal and k = 3 for t. AICc = -2 log L + 2k + 2k(k+1)/(n-k-1); lower is better. The parameter count includes location, scale, and, for the t distribution, degrees of freedom.')
rows=s['p4']['margin_fits'];tbl(['Asset / model','Location','Scale','nu','AICc'],[[r['asset']+' / '+r['family']+(' *' if r['selected'] else ''),f"{r['mu']:.7f}",f"{r['scale']:.7f}",'--' if r['nu'] is None else ('normal limit' if r['nu']>1e8 else f"{r['nu']:.4f}"),f"{r['AICc']:.3f}"] for r in rows],[130,95,95,95,95])
p('* Selected: normal for X1, t for X2, and t for X3. X1\'s t fit runs toward infinite degrees of freedom (numerically about 2.55e12), giving the same likelihood as the normal but an extra parameter penalty; that huge finite estimate has no substantive interpretation. Normal wins X1 by 2.012 AICc points; t wins X2 by 109.691 and X3 by 62.324.',True)
h('(d) Copula estimation')
p('Transform each return through its selected fitted CDF: u_ij = F_j(x_ij). Estimate Kendall tau for each pair, then R_ij = sin(pi tau_ij/2). The resulting R is positive definite and requires no repair. The same R is used for both copulas:')
mat('p4_kendall_R')
p('For each observation, the copula log density is the multivariate normal or t log density of the transformed quantiles minus the sum of their univariate log densities. The t degrees of freedom maximize this likelihood over [2,100], using a profile grid followed by local refinement. The optimum is interior.')
tbl(['Copula','Log likelihood','nu','k','AICc','BIC'],[[r['model'],f"{r['ll']:.4f}",'--' if r['nu'] is None else f"{r['nu']:.6f}",r['k'],f"{r['AICc']:.4f}",f"{r['BIC']:.4f}"] for r in s['p4']['comparison']],[80,100,80,40,105,105])
p('<b>Counting convention:</b> these are conditional copula criteria, following Week05/choosing_between_copulas.jl: margins and rank-based R are held fixed, so k = 0 for Gaussian and 1 for t. BIC = -2 log L + k log n. Counting the three shared estimated correlations instead would give k = 3 and 4, changing absolute criteria but not the conclusion (BIC difference is identical; AICc difference changes by only about 0.012). Marginal parameters are not counted again in this conditional comparison.')
p('These fits agree with the predicted marginal choices and copula preference. The Gaussian copula is not equivalent to normal margins: both simulations below use exactly the same selected normal/t margins, isolating the effect of the dependence model.')

page();title('4. Copulas - Simulation and tail comparison')
h('(e) Portfolio VaR and ES')
p('Generate correlated standard normals with covariance R. For t, divide every component of a day by the same sqrt(chi-square(nu)/nu), then apply the t CDF; for Gaussian apply the normal CDF. Convert uniforms through the selected marginal inverse CDFs and sum $1m times each return. Fitted means are retained because this problem does not impose zero expected returns. Both models use 100,000 draws and common Gaussian innovations to reduce comparison noise.')
tbl(['Model','VaR 5%','ES 5%','VaR 1%','ES 1%'],[[r['model'],money(r['VaR5']),money(r['ES5']),money(r['VaR1']),money(r['ES1'])] for r in s['p4']['risk']],[90,105,105,105,105])
p('Historical measures use the original 1,000 portfolio P&amp;Ls: only 50 observations enter 5% ES and 10 enter 1% ES. Simulation values are reproducible estimates, not exact integrals; especially the far-tail ES values remain subject to Monte Carlo and model-estimation uncertainty.')
h('(f-g) Joint tails implied over 1,000 days')
tbl(['Model','Pair','Both lower 2.5%','Both upper 2.5%'],[[r['model'],r['pair'],f"{r['lower']:.2f}",f"{r['upper']:.2f}"] for r in s['p4']['tail']],[150,100,130,130])
p('Simulation tail counts use uniform thresholds 0.025 and 0.975, the model-implied marginal percentiles, then multiply the simulated frequency by 1,000. Observed counts use empirical ranks. Fractional implied counts are expectations estimated by simulation; they are not integer observations.')
p('The t copula increases expected joint-tail days from roughly 4-6 to 7-10. It better reflects the overall tail clustering, particularly the large upper-tail counts, but does not match every cell: Gaussian is closer to the observed lower X1-X2 count of 6, while neither symmetric model captures the observed upper/lower imbalance exactly. Differences between simulated lower and upper counts arise from Monte Carlo variation, since the copulas themselves are symmetric.')
g,t=s['p4']['comparison'];p(f"The t gains {t['ll']-g['ll']:.3f} log-likelihood units and wins by {g['AICc']-t['AICc']:.3f} AICc points and {g['BIC']-t['BIC']:.3f} BIC points. Both penalties are small relative to the likelihood improvement. Its strong model-selection result is not a guarantee that every observed tail count or risk estimate will be matched.")

page();title('4. Copulas - What the winning score means')
h('(h) Which risk number moves most?')
g,t,_=s['p4']['risk'];tbl(['Measure','t minus Gaussian','Relative change'],[[key,money(t[key]-g[key]),f"{100*(t[key]/g[key]-1):+.2f}%"] for key in ['VaR5','ES5','VaR1','ES1']],[170,170,170])
p('The largest movement is <b>1% ES</b>, up about $5,975 or 6.29%. It averages the deepest losses, where the shared t scaling shock makes simultaneous extremes more likely. The 5% VaR moves only about -$690 (1.38%): the marginal distributions and rank-based correlation are unchanged, and this moderate quantile is much less sensitive to asymptotic tail dependence. The t copula redistributes probability across the joint distribution; it does not raise all lower-tail quantiles uniformly. The small 5% difference should not be overinterpreted beyond simulation precision.')
h('(i) Daily likelihood contributions')
a=s['p4'];tbl(['Observation group','Days','Sum of log-likelihood gain'],[['All',1000,f"{a['total_gain']:.6f}"],['No rank outside 5%-95%',a['central_days'],f"{a['central_gain']:.6f}"],['At least one rank outside',1000-a['central_days'],f"{a['total_gain']-a['central_gain']:.6f}"]],[255,65,190])
p(f"For each day, results/p4_daily_loglik.csv records ell_t minus ell_Gaussian. Using empirical ranks to define each series' outer 5% on each side, {a['central_days']} central days contribute {a['central_gain']:.3f}, or {100*a['central_fraction']:.2f}% of the total gain. The remaining 216 days contribute 51.759. Thus about half the improvement comes from days with no marginal tail observation at all.")
p(f"As a definition sensitivity check, using fitted-CDF thresholds instead gives {a['fitted_central_days']} central days and a gain of {a['fitted_central_gain']:.3f} (49.70% of the total). This does not change the interpretation. AICc/BIC score density fit over every observation and penalize parameter count; they are not dedicated scores for joint crashes, VaR, or ES. A t copula changes central concentration as well as extremes, so a large likelihood win need not translate into a large change in 5% VaR.")
h('(j) Asymptotic tail dependence')
p(f"The most correlated pair is {a['tail_pair']}, with latent rho = {a['rho']:.6f}. For a t copula, lower and upper tail dependence equal 2 T_(nu+1)(-sqrt((nu+1)(1-rho)/(1+rho))). With nu = {a['nu']:.6f}, the coefficient is <b>{a['tail_dependence']:.6f}</b>. For a nondegenerate Gaussian copula it is <b>0</b>.")
p('This coefficient is the limiting conditional probability of one component being extreme given that the other is extreme, as the tail probability shrinks to zero. It is not the unconditional probability of a joint 2.5% tail event. A zero Gaussian limit is compatible with its positive finite-threshold counts shown on the previous page.')

page();title('5. Model simulation and residual correlation')
h('Predict (a): signs from the variance formula')
p('With a shared omitted industry factor, residual covariance should be positive. Portfolio variance includes 2 w_A w_B Cov(e_A,e_B). For P1 the weight product is positive, so treating residuals as independent removes a positive contribution and understates VaR. For P2 the weight product is negative; removing that covariance removes a hedge and overstates VaR.')
h('Fit (b): regressions and covariance convention')
p('The 501 prices produce 500 arithmetic returns. Both OLS regressions include an intercept. Alpha and beta are reported below. Residual standard deviations use sample covariance with denominator n-1, consistently with direct stock-return covariance. The usual regression residual standard errors (n-2) are also shown to distinguish the conventions.')
tbl(['Stock','Alpha','Beta','Residual sd (n-1)','Residual SE (n-2)'],[[r['asset'],f"{r['alpha']:.9f}",f"{r['beta']:.6f}",f"{r['residual_sd']:.7f}",f"{r['regression_residual_se']:.7f}"] for r in s['p5']['regressions']],[50,115,95,125,125])
p(f"Residual correlation is <b>{s['p5']['residual_correlation']:.6f}</b>, a substantial positive dependence remaining after the market factor. The residual covariance matrix is:")
mat('p5_residual_cov')
h('(c-d) Simulated and delta-normal VaRs')
p('Simulate a zero-mean normal market return and bivariate zero-mean normal residuals independent of the market. Use either the full residual covariance or its diagonal only. Form stock returns as beta times market plus residual. Estimated intercepts are deliberately excluded because the problem assumes zero expected returns. Use the same base normal draws for both covariance assumptions.')
tbl(['Residual model','Portfolio','Simulated VaR','Analytic model VaR','Direct delta-normal'],[[r['assumption'],r['portfolio'],money(r['simulated_VaR']),money(r['analytic_VaR']),money(r['delta_normal_VaR'])] for r in s['p5']['risk']],[126,54,105,115,110])
p('The analytic model column evaluates 1.644854 sqrt(w\'C_model w). Direct delta-normal uses the sample covariance of A and B returns and the same zero mean. It is repeated across assumptions for comparison; it does not change when the residual shortcut is applied.')
h('Reconcile (e): direction and exposure')
p('Both directions match the prediction. Analytically, independence reduces P1 VaR by $5,921 (11.74%) and raises P2 by $12,673 (80.04%). The covariance term has equal absolute size and opposite sign in the two variances, but P2 has much smaller baseline risk. Its market betas almost cancel (1.0353 - 0.9354 = 0.0999), so residual co-movement is central to the hedge. P2 is more exposed to this assumption in relative VaR and also in dollar VaR here.')

page();title('5. Reconciliation and reproducibility')
h('(f) Why full-residual simulation agrees with delta-normal')
p('OLS with an intercept makes each residual sample mean zero and makes the residual orthogonal to the market regressor. Therefore, using the same n-1 covariance denominator throughout, the sample stock covariance decomposes exactly as:')
p('<b>S_stocks = beta beta\' s_market^2 + S_residual.</b>')
p('The largest entrywise difference between the reconstructed and direct covariance is about 1.08e-19. Under the joint-normal model, OLS residual orthogonality also corresponds to independence between market and residuals, so the simulated return model is a joint normal with exactly this covariance. Every linear portfolio consequently has the same theoretical zero-mean VaR as delta-normal.')
p('Simulation estimates differ slightly: full-residual P1 is $50,350 versus analytic $50,415, and P2 is $15,860 versus $15,834. These are Monte Carlo quantile differences, not a covariance disagreement. The exact covariance identity is checked by an assertion in the analysis and independently in the tests.')
p('Agreement can stop if the marginals or innovations are not jointly normal, if conditional volatility changes, if residuals depend nonlinearly on the market, or if the portfolio has nonlinear payoffs. Zero sample covariance alone does not imply independence outside a joint-normal model. Changing sample windows, using inconsistent covariance denominators, discarding residual correlation, or including means in only one method can also break the numerical agreement. The OLS covariance identity itself remains algebraically true on the same sample even if normality is false; what then fails is the implication that covariance determines tail risk.')
h('Implementation and reproducibility')
p('The quant_risk package implements covariance repair, descriptive moments, distribution fitting, information criteria, historical VaR and ES, and copula estimation and simulation. The five analyses in run_analysis.py use these shared functions. Installation, execution, and test commands are provided in README.md.')
p('Run the commands in README.md from Assignment2: create a virtual environment, install with <font name="Courier">python -m pip install -e .</font>, execute <font name="Courier">python run_analysis.py</font>, run <font name="Courier">python -m unittest discover -s tests -v</font>, and rebuild this PDF with <font name="Courier">python build_report.py</font>. CSV outputs preserve more precision than the rounded written tables. Fixed seeds make the simulation reproducible in the recorded environment; see environment_versions.json.')
h('Sources and conventions')
p('Source data: the five problem CSVs in the course repository, Assignments/Assignment2. Questions: assignment.qmd / Assignment 2.pdf. Course implementation references: library/ewCov.jl (normalized EW covariance), library/RiskStats.jl (loss-positive empirical risk), library/copula.jl and Week05/choosing_between_copulas.jl (copula construction and conditional information criteria). The covariance module adapts the existing functional-tests-1/risk_management implementation in this repository. No external market data are used.',True)
p('Historical quantiles use the inverse empirical CDF. At every alpha and sample size used here, n alpha is an integer, matching the course order-statistic cutoff. The reusable ES function averages exactly alpha probability mass, also defining sensible behavior for ties or fractional tail sizes. Matrix repair tolerances, return scaling, fitting parameter counts, and covariance denominators are documented in code.',True)

def footer(canvas,doc):
    canvas.saveState();canvas.setStrokeColor(colors.HexColor('#cfdae0'));canvas.line(42,39,570,39);canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#51636d'));canvas.drawString(42,26,'FinTech 545 | Assignment 2');canvas.drawRightString(570,26,str(doc.page));canvas.restoreState()

if __name__=='__main__':
    path=BASE/'Assignment2_Responses.pdf'
    SimpleDocTemplate(str(path),pagesize=(612,792),rightMargin=42,leftMargin=42,topMargin=38,bottomMargin=52,title='Assignment 2 - Covariance, VaR, and Copulas',author='Chuyu Qin').build(story,onFirstPage=footer,onLaterPages=footer)
    print(path)
