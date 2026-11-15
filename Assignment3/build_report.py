"""Build the written submission from saved CSV/JSON results; no model reruns.

Narrative interpretations refer to the supplied data. If inputs change, review
this narrative as well as regenerating tables. Uses ReportLab, not LaTeX.
"""
from pathlib import Path
import csv,json,html,math
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.utils import ImageReader
B=Path(__file__).resolve().parent;R=B/'results';s=json.loads((R/'summary.json').read_text())
styles=getSampleStyleSheet();blue=colors.HexColor('#12354a')
styles.add(ParagraphStyle(name='BodyA',fontName='Helvetica',fontSize=9.5,leading=12.5,spaceAfter=7))
styles.add(ParagraphStyle(name='SmallA',fontName='Helvetica',fontSize=8,leading=10,spaceAfter=4))
styles.add(ParagraphStyle(name='TitleA',fontName='Helvetica-Bold',fontSize=19,leading=23,spaceAfter=12,textColor=blue))
styles['Heading2'].textColor=blue;styles['Heading2'].fontSize=12;styles['Heading2'].leading=15;styles['Heading2'].spaceBefore=9;styles['Heading2'].spaceAfter=7
story=[]
def p(t,small=False):story.append(Paragraph(t,styles['SmallA' if small else 'BodyA']))
def h(t):story.append(Paragraph(t,styles['Heading2']))
def title(t):story.append(Paragraph(t,styles['TitleA']))
def page():story.append(PageBreak())
def rows(name):
    with open(R/(name+'.csv')) as f:return list(csv.DictReader(f))
def n(x):return float(x)
def fmt(x,d=4):
    try:
        v=float(x)
        return 'N/A' if math.isnan(v) else f'{v:,.{d}f}'
    except (ValueError,TypeError):return 'N/A' if x=='' else str(x)
def pc(x,d=3):return fmt(100*n(x),d)+'%'
def money(x,d=2):return ('$' if n(x)>=0 else '-$')+fmt(abs(n(x)),d)
def tbl(headers,data,widths=None):
    cell=[[Paragraph(html.escape(str(v)),styles['SmallA']) for v in row] for row in [headers]+data]
    t=Table(cell,colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e2ecf1')),('VALIGN',(0,0),(-1,-1),'TOP'),('BOTTOMPADDING',(0,0),(-1,-1),2),('TOPPADDING',(0,0),(-1,-1),2),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#829aa8')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f5f7f9')])]))
    story.extend([t,Spacer(1,8)])
def fig(name,width=505):
    path=B/'figures'/(name+'.png');w,hh=ImageReader(str(path)).getSize();story.extend([Image(str(path),width=width,height=width*hh/w),Spacer(1,8)])
short={'CoveredCall':'Covered call','ShortStraddle':'Short straddle','RiskReversal':'Risk reversal','HedgedShortCall':'Hedged short call','Unconstrained':'Unconstrained','Long_only':'Long only','PUTW_cap20':'PUTW cap 20%'}

# Page 1
title('Assignment 3: Options, Risk, and Portfolios')
p('FinTech 545 | Chuyu Qin',True)
p('Returns are arithmetic and expressed as decimals in code. VaR = minus the lower P&amp;L quantile; ES = minus the mean of the worst alpha probability mass. A positive number means a loss. Historical ES integrates fractional boundary weight; no risk number is floored at zero. Variance uses n-1; skewness and excess kurtosis use empirical central moments.')
p('Predictions were recorded in predictions.md using the pre-fit diagnostics in preview.py before the main fits. Calendar time is used in Problems 1-2; Problem 3 uses 255 trading days per year. Problems 4-5 report return risk, not dollar risk unless explicitly labeled.',True)
h('1. Dirty option chain - Predict (a-c)')
p('<b>(a)</b> T = 35/365 years. Define gap = C_mid - P_mid - [S exp(-qT) - K exp(-rT)], with S = 182.40, r = 0.0425, q = 0.006. The requested flagging band is the <b>sum of the full call and put spreads</b>. I flag only |gap| &gt; band. This is the stated conservative rule, rather than substituting a half-spread band.')
tbl(['Strike','Parity gap ($)','Band ($)','Flag'],[[fmt(a['Strike'],0),fmt(a['gap'],6),fmt(a['band'],2),'YES' if a['flag']=='True' else ''] for a in rows('p1_parity')],[70,160,150,130])
p('Only <b>K = 175</b> is flagged: gap = +$0.888179 exceeds $0.47. At least one side is inconsistent with a common European pricing model; parity alone cannot identify which quote is wrong. Ignoring trading frictions beyond the quoted spreads, selling the rich call-minus-put and buying the matching forward locks in the discrepancy. The plots will help localize it.')
p('<b>(b)</b> For a fixed strike and expiry, GBSM gives C(sigma) - P(sigma) = S exp(-qT) - K exp(-rT), independent of sigma. With valid interior quotes and positive vega, each inversion is unique. Thus exact parity forces the same call and put IV. A small nonzero gap can still cause visibly different IVs where vega is tiny.')
p('<b>(c)</b> I expect the least stable IVs in the far wings: vega is small, so d sigma / d price = 1/vega is large. The 215, 220, and 225 calls all quote $0.05/$0.10; identical tiny prices at successively farther strikes suggest a quote floor, so I predict an artificial rising call wing. Deep ITM quotes also have little time value and can cross the zero-volatility lower bound at their bids.')

# Page 2
page();title('1. Implied volatility - Fit (d)')
p('The option pricer is my own generalized BSM formula: V = z[S exp((b-r)T) N(z d1) - K exp(-rT) N(z d2)], where z = +1 for calls and -1 for puts, b = r-q, d1 = [log(S/K)+(b+sigma^2/2)T]/(sigma sqrt(T)), and d2 = d1-sigma sqrt(T). IV uses a bracketed root of V(sigma)-quote. No option-pricing library is called.')
ivs=rows('p1_iv');lookup={(n(a['Strike']),a['Type']):a for a in ivs};strikes=sorted(set(n(a['Strike']) for a in ivs))
tbl(['K','Call bid IV','Call mid IV','Call ask IV','Put bid IV','Put mid IV','Put ask IV'],[[fmt(k,0)]+[pc(lookup[k,t]['IV_'+side],2) if lookup[k,t]['IV_'+side] else 'N/A' for t in ['Call','Put'] for side in ['Bid','Mid','Ask']] for k in strikes],[38,79,79,79,79,79,79])
p('<b>All 36 mids and all 36 asks invert.</b> The call bids at 140, 145, 150, 155 and the put bids at 205, 210, 215, 220, 225 do not: they lie below max[S exp(-qT)-K exp(-rT), 0] for calls, or max[K exp(-rT)-S exp(-qT), 0] for puts. No nonnegative volatility can produce those bids. I report N/A rather than force a root or replace the input quote. An invalid bid is a mathematical pricing-bound failure; a far-wing mid can still be valid but economically unreliable.')
fig('p1_iv',440)
p('Shaded regions are the IV range implied by valid bid/ask endpoints. A missing lower edge reflects an invalid bid; it does not mean zero uncertainty. The quote-level statuses, prices, and full-precision roots are in results/p1_iv.csv.',True)

# Page 3
page();title('1. Vega and quote quality - Reconcile (e-h)')
p('<b>(e)</b> Vega = S exp(-qT) phi(d1) sqrt(T), evaluated separately at each quote\'s midpoint IV. Vega below is dollars per +1.00 volatility (100 percentage points). The last columns report the <b>local change in volatility percentage points per $0.01</b>, equal to 100(0.01/vega) = 1/vega. This is a derivative, not an exact finite-price inversion.')
tbl(['K','Call vega','Put vega','Call vol pts / cent','Put vol pts / cent'],[[fmt(k,0)]+[fmt(lookup[k,t]['vega'],4) for t in ['Call','Put']]+[fmt(lookup[k,t]['vol_points_per_cent'],4) for t in ['Call','Put']] for k in strikes],[50,100,100,130,132])
a=s['p1'];p(f"<b>(f)</b> The strike-175 put is the conspicuous dip: put IV = 21.1620%, versus call IV = 26.0457%. The +${a['gap']:.6f} parity gap predicts a call-minus-put IV difference of {100*a['linear_gap']:.4f} percentage points using the put's vega ({a['vega_put']:.4f}). The actual difference is {100*a['iv_gap']:.4f} points. Starting at the call's higher vega gives 4.7139 points; the exact gap lies between these local approximations because vega changes over the interval. The smooth surrounding call curve and depressed put suggest the put quote is stale or low, though that diagnosis needs quote verification.")
p('<b>(g)</b> Above 210 the call mids are all $0.075 and IV rises from 23.8225% at 215 to 26.6888% at 220 and 29.4473% at 225. One cent moves IV by about 0.44-0.52 points. Bid/ask IVs are 22.5619-24.8267%, 25.3138-27.7811%, and 27.9646-30.6226% respectively. The mechanically rising quote floor affects the whole range, not just the mids; it is poor evidence of market tail demand. Parity does not flag it because the gaps ($0.0355, $0.0602, $0.0698) are tiny compared with the deep-ITM put spreads and the requested full-spread bands ($1.05, $1.20, $1.36).')
p('<b>(h)</b> Excluding the strike-175 inconsistency and the floored call wing, the smile is mainly a <b>downward equity skew</b>: about 39% at strike 140, 25% near 180, and 22% around 210-225 using the cleaner puts in the upper wing. Lower-strike downside protection trades at higher implied volatility, consistent with stronger demand for crash insurance and/or a downside risk premium. This is a price-based interpretation, not proof of a unique physical return distribution. My wing and parity predictions were supported.')

# Page 4
page();title('2. American options and dividends - Predict / Fit')
p('<b>(a) Predict.</b> The call-exercise condition D &gt; K[1-exp(-r(T-t))] gives thresholds $0.891948 at t=0.10 and $0.224495 at t=0.40. The first dividend ($0.25) fails; the second ($1.50) passes. Thus exercise can be optimal just before the second dividend, provided spot is sufficiently high. Passing the condition does not say exercise is optimal at every spot.')
p('<b>(b) Predict.</b> Dividends lower the European and American calls and raise the European and American puts. I predict a larger call early-exercise premium and a smaller put premium. A call may be exercised to capture the large second dividend; a put benefits from waiting through a downward ex-dividend move.')
h('(c) Pricing method and values')
p('Use CRR u = exp(sigma sqrt(dt)), d = 1/u, p = [exp(r dt)-d]/(u-d), discount exp(-r dt), with S = 48.50, K = 50, T = 0.5, r = 4.5%, sigma = 30%. At a dividend, stock jumps to max(S-D,0). Every node starts an independent shifted subtree; no global recombination or interpolation is imposed. Exercise is checked before and after the event. The zero-stock boundary is absorbing. Both dates fall exactly on the grid because N is a multiple of five.')
p('Baseline results use <b>N = 200</b> (dividend steps 40 and 160). Dollar values are per option on one share. GBSM is the reported no-dividend European benchmark; the extra tree column makes tree-only premium comparisons consistent.',True)
pr=rows('p2_prices')
tbl(['Option','No-div GBSM','No-div tree','Cash-div tree','PV-adjusted GBSM'],[[a['style']+' '+a['Type']]+[fmt(a[k],6) for k in ['no_div_gbsm','no_div_tree','div_tree','pv_adjusted_gbsm']] for a in pr],[115,95,95,100,105])
p('The PV shortcut uses S* = 48.50 - 0.25 exp(-0.045 x 0.10) - 1.50 exp(-0.045 x 0.40) = $46.777881. It gives European call $3.072293 and put $5.181974, each about $0.102881 below the cash-dividend tree. This is a model approximation, not an alternative exact discrete-dividend tree: it treats the adjusted initial spot as if it followed a single multiplicative diffusion. In the actual cash-jump model, volatility acts on the pre-dividend stock and each subtraction shifts the later branches.')
h('(g) Early-exercise premiums, using matched trees')
prems=[]
for typ in ['Call','Put']:
    eu=next(a for a in pr if a['Type']==typ and a['style']=='European');am=next(a for a in pr if a['Type']==typ and a['style']=='American')
    prems.append([typ,fmt(n(am['no_div_tree'])-n(eu['no_div_tree']),6),fmt(n(am['div_tree'])-n(eu['div_tree']),6)])
tbl(['Option','No-dividend premium ($)','Cash-dividend premium ($)'],prems,[130,190,190])
p('All four value-direction predictions hold. The call premium rises from zero to $0.292370, while the put premium falls from $0.130115 to $0.086280. If the analytic no-dividend European benchmark is used instead, the put premium is $0.125450; that small difference is European tree discretization. I use matched trees to isolate exercise value.')

# Page 5
page();title('2. Convergence, Greeks, and exercise - Fit / Reconcile')
h('(d) American cash-dividend tree convergence')
tbl(['N','Call ($)','Put ($)'],[[a['N'],fmt(a['Call'],6),fmt(a['Put'],6)] for a in rows('p2_convergence')],[90,210,210])
p('Neighbouring grids oscillate: 100 to 105 moves the call by -$0.007533 and the put by -$0.011076. At 400/405 the gap is still $0.001407/$0.002221. The values settle around $3.469/$5.371, but convergence is not monotonic. Reporting many decimals at one N would exaggerate accuracy. The baseline Greek calculations intentionally keep N fixed at 200.',True)
h('(e, h) Central finite differences on the N = 200 tree')
gg=rows('p2_greeks');tbl(['Bump h ($)','Call delta','Call gamma','Put delta','Put gamma'],[[fmt(a,3)]+[fmt(next(z for z in gg if z['Type']==t and abs(n(z['bump'])-a)<1e-10)[k],7) for t in ['Call','Put'] for k in ['delta','gamma']] for a in [1,.1,.01,.001]],[85,105,105,105,110])
p('Delta = [V(S+h)-V(S-h)]/(2h), gamma = [V(S+h)+V(S-h)-2V(S)]/h^2. <b>Gamma is unstable for both options</b>, especially the call: its apparent gamma collapses from 0.0786 at h=0.1 to essentially zero at h=0.001. A fixed tree is piecewise linear in initial spot, with kinks from payoff and exercise boundaries. Tiny bumps either sit on one linear piece (zero curvature) or cross a kink (a large, grid-dependent quotient). Put gamma is also irregular; the last two values, 0.0178 and 0.0224, do not establish convergence. Shrinking h at fixed N cannot recover the smooth limiting gamma. Delta is much more stable near the smallest bumps. A practical estimate needs bump and grid sensitivity together.')
h('(f) Sensitivity to each cash dividend ($ value / $ dividend)')
tbl(['Option','dV/dD1','dV/dD2'],[[a['style']+' '+a['Type'],fmt(a['sensitivity'],6),fmt(next(z for z in rows('p2_dividend_sensitivity') if z['Type']==a['Type'] and z['style']==a['style'] and z['dividend']=='2')['sensitivity'],6)] for a in rows('p2_dividend_sensitivity') if a['dividend']=='1'],[210,150,150])
p('These are central differences with a $0.001 dividend bump and fixed payment dates. The European call loses about $0.426/$0.372 per dollar of D1/D2. The American call loses $0.484/$0.084: it cannot optimally exercise for the small first dividend, but many high-spot branches exercise before the large second one, protecting their value from additional D2. Puts gain from either downward stock jump. As with the spot Greeks, these local tree derivatives inherit discretization.',True)
p(f"<b>(g, i)</b> Immediately before t=0.40, solve S-50 = V_call(S-1.50, 0.10 years remaining). The N=200-consistent final 40-step subtree gives a boundary <b>S = ${s['p2']['boundary']:.6f}</b>; exercise is optimal above it (indifferent at it). Analytic no-more-dividends continuation gives ${s['p2']['boundary_smooth']:.6f}, a useful grid check. The exercise decision explains the American call's much smaller sensitivity to the second dividend, and supports the prediction in (a). The European call cannot use that protection.")

# Page 6
page();title('3. Option book risk - Predict / Fit')
p('The last ABC price is $99.12. Holdings count shares or one-share options; there is no 100-share contract multiplier. Option maturity starts at 30/255 years and becomes 20/255 after the horizon. Implied volatilities are held fixed. Greeks below are position-weighted; theta is dollars per year of elapsed time, so the 10-day theta contribution is theta x 10/255.')
gr=rows('p3_greeks')
tbl(['Book','Initial value ($)','Delta','Gamma','Theta / year','P&L: flat spot'],[[short[a['Portfolio']],fmt(a['value'],2),fmt(a['delta'],4),fmt(a['gamma'],4),fmt(a['theta'],2),fmt(a['flat_spot_pnl'],2)] for a in gr],[110,85,72,72,85,88])
p('<b>(a) Predict.</b> The short straddle and hedged short call rank jointly worst for relative delta-normal error: their deltas are near zero but their negative gammas are large. Delta-normal should substantially understate their losses. Next is the covered call: negative gamma makes its downside loss worse than the local linear estimate. The risk reversal has the least expected error; positive local gamma suggests a tentative overstatement. This last direction is uncertain because exposures change across its two strikes. Absolute errors need not have the same ranking as relative errors.')
p('<b>(b) Predict.</b> All four initial thetas are positive, so I predict a gain at unchanged spot. The exact flat-spot P&amp;Ls shown above confirm that sign. I expect the largest 5% ES/VaR ratios in the two near-neutral short-gamma books because large moves in either direction hurt them. For a zero-mean normal P&amp;L, ES/VaR is 1.254 at 5% but only 1.146 at 1%; 1.25 is not a universal reference for all confidence levels or nonzero means.')
h('(c) Return moments and distribution selection')
m=rows('p3_moments')[0]
tbl(['n','Mean','Variance','Skewness','Excess kurtosis'],[[m['n'],fmt(m['mean'],9),fmt(m['variance'],9),fmt(m['skew'],6),fmt(m['excess_kurtosis'],6)]],[55,120,120,105,112])
tbl(['Model','Fitted location','Scale / MLE sd','nu','Implied sd','AICc'],[[a['model'],fmt(a['mu'],9),fmt(a['scale'] if a.get('scale') else a['sd'],9),fmt(a.get('nu',''),6),fmt(a['sd'],9),fmt(a['AICc'],4)] for a in rows('p3_fits')],[85,95,100,70,90,72])
p('The Student t wins by 28.4227 AICc units, consistent with excess kurtosis 2.4274 and nearly zero skew. Both fits are maximum likelihood; normal scale uses n, whereas the sample moment variance uses n-1. The t scale 0.013506 is not its standard deviation (0.016694). AICc = -2 log L + 2k + 2k(k+1)/(n-k-1), with k = 2 and 3. For forecasting, I remove each fitted location to impose zero daily expected return; parameters above describe the original fit.')
h('(d) Ten-day delta-normal risk ($)')
dn=rows('p3_delta_normal')
tbl(['Book','5% VaR','5% ES','1% VaR','1% ES'],[[short[name]]+[fmt(next(a for a in dn if a['Portfolio']==name and n(a['alpha'])==al)[k],2) for al in [.05,.01] for k in ['VaR','ES']] for name in short if name in {a['Portfolio'] for a in gr}],[150,90,90,90,92])
p('Dollar P&amp;L sd = |book delta| x 99.12 x sample daily sd x sqrt(10). This method includes neither theta nor gamma. The daily sample sd is 1.667428%; the normal MLE sd differs slightly.',True)

# Page 7
page();title('3. Full revaluation and second-order approximation')
p('<b>(e)</b> Simulate 500,000 independent 10-day paths using zero-location normal and Student t daily arithmetic returns, compound S10 = S0 product(1+r_i), and reprice every position at 20 trading days remaining. Seed = 5453026; common uniforms couple the two distributions. Since daily means are zero, expected compounded spot is S0 under independence. No observed simulated daily return is below -100%; such a draw triggers an error rather than a silent clipping convention. The fitted t does have unbounded theoretical support, a limitation of this arithmetic-return model.')
full=rows('p3_full_revaluation');ap=rows('p3_delta_gamma_theta')
for model in ['Normal','Student_t']:
    h(model.replace('_',' ')+' full revaluation: dollars')
    tbl(['Book','Mean P&L','5% VaR','5% ES','1% VaR','1% ES'],[[short[name],fmt(next(a for a in full if a['model']==model and a['Portfolio']==name)['mean'],2)]+[fmt(next(a for a in full if a['model']==model and a['Portfolio']==name and n(a['alpha'])==al)[k],2) for al in [.05,.01] for k in ['VaR','ES']] for name in [a['Portfolio'] for a in gr]],[130,78,76,76,76,76])
p('Monte Carlo uncertainty is reported in results/p3_mc_uncertainty.csv using ten disjoint batches. Mean standard errors are about $0.11-$0.53; 1% ES standard errors are about $2.00-$4.69. Differences of a dollar or two should not be overinterpreted. These assess simulation noise conditional on fitted parameters, not parameter or model uncertainty.',True)
h('(g) Does delta-gamma-theta reproduce the tail?')
p('Evaluate delta dS + 0.5 gamma (dS)^2 + theta (10/255) on the <b>same paths</b>. Adding gamma makes the near-neutral books risky, and theta captures their carry. It does not exactly reproduce full repricing because the 10-day move and one-third reduction in remaining maturity are large enough for Greeks to change.')
tbl(['Book / model','Approx 5% VaR','Full 5% VaR','Approx 1% ES','Full 1% ES'],[[short[name]+' / '+('N' if model=='Normal' else 't')]+[fmt(next(a for a in source if a['model']==model and a['Portfolio']==name and n(a['alpha'])==al)[k],2) for source,al,k in [(ap,.05,'VaR'),(full,.05,'VaR'),(ap,.01,'ES'),(full,.01,'ES')]] for model in ['Normal','Student_t'] for name in [a['Portfolio'] for a in gr]],[160,88,88,88,88])
p('For example, t short-straddle 1% ES is $963.25 in the expansion versus $841.86 in full revaluation. The t risk-reversal approximation goes the other way, $797.94 versus $928.53: the local positive gamma does not describe the far-left short-put tail. Full repricing is needed for the numbers, even though the expansion explains the exposure.')

# Page 8
page();title('3. P&amp;L shapes - Reconcile (f-j)')
fig('p3_pnl',425)
p('<b>(f, g)</b> Each solid curve includes ten days of decay; each dashed line is delta-only and passes through zero at the initial spot. The nearly horizontal short-straddle and hedged-call lines miss the downward curvature. At fixed spot, delta-only is zero while the options earn decay. For the covered call, upside is capped while stock downside remains. The risk reversal changes curvature as spot crosses its put and call strikes.')
p('<b>(h)</b> The unchanged-spot sign prediction holds for every book. The simulated means are also positive: about $6.6-$6.8 covered call, $20.0-$20.4 short straddle, $8.6 risk reversal, and $18.3-$18.5 hedged short call. The short-gamma books earn time decay in exchange for losses on sufficiently large moves. Their option-implied volatilities exceed the fitted realized daily vol annualized (about 26.6%), leaving positive average carry under these forecasting distributions. Positive mean is compensation for nonlinear downside, not proof the positions are low risk.')
p('<b>(i)</b> Switching normal to t barely changes 5% VaR for covered call ($677.42 to $676.77) or risk reversal ($489.65 to $489.18), and moves it modestly for the neutral books. Ten-day aggregation and similar daily variances make central quantiles similar. Tail severity moves more: t-minus-normal 1% ES is about +$69.11 covered call, +$94.25 straddle, +$61.52 risk reversal, and +$47.12 hedged call. Those are roughly 6.0%, 12.6%, 7.1%, and 12.9% increases. The neutral short-gamma books show the largest proportional sensitivity because extreme moves from either side become losses. The 5% cutoff need not shift much when mass farther beyond it increases.')
p('The expected ES/VaR inflation appears: at 5%, the t straddle ratio is 1.617 and hedged-call ratio 1.650, compared with covered call 1.320 and risk reversal 1.345. At 1%, the t ratios are 1.324 and 1.333 for the neutral books. The comparison must use the correct 1% normal reference of 1.146, not 1.25.')
p('My relative-error ranking is broadly supported. The tentative risk-reversal overstatement holds at 5% ($527.75 delta-normal versus $489.18 t full), but fails at 1% ($746.40 versus $760.67), especially for ES ($855.12 versus $928.53). I had relied too much on local positive gamma; sufficiently far downside leaves a short put and loses that local protection.')
p('<b>(j) Reply to the risk manager (two sentences).</b> The hedged short call is only approximately delta neutral today; it still has gamma of -4.046 dollars per squared dollar of spot and earns theta by selling convexity. Its 10-day 1% loss risk is about $309.53 VaR and $412.63 ES under the fitted t model, so calling it flat conceals substantial tail exposure.')

# Page 9
page();title('4. A Sharpe ratio that sells insurance - Predict / Fit')
p('Use 240 monthly total-return observations, monthly risk-free return 0.03/12 = 0.0025, annual mean = 12 x monthly mean, annual sd = sqrt(12) x monthly sd, and annual Sharpe = sqrt(12)(mean-rf)/sd. All portfolio weights sum to one; unconstrained weights may be negative. No risk-free weight is included in the four-asset tangency portfolio.')
h('(a) Moments and annualized Sharpe')
assets=rows('p4_assets')
tbl(['Asset','Mean / mo','Variance','Skewness','Excess kurtosis','Ann. Sharpe'],[[a['asset'],pc(a['mean']),fmt(a['variance'],8),fmt(a['skew'],4),fmt(a['excess_kurtosis'],4),fmt(a['sharpe'],4)] for a in assets],[60,90,95,85,100,82])
a=next(a for a in assets if a['asset']=='PUTW')
p(f"PUTW's Sharpe 0.7834 looks unusually attractive given skewness -6.5254 and excess kurtosis 59.8505. Its worst month, {a['worst_month']}, returns {pc(a['worst'],4)} and contributes {pc(a['variance_share_worst'],2)} of its full-sample squared deviations and {pc(a['fourth_share_worst'],2)} of fourth deviations. Removing just that month raises the monthly mean from {pc(a['mean'],4)} to {pc(a['mean_without_worst'],4)}, reduces sd from {pc(a['sd'],4)} to {pc(a['sd_without_worst'],4)}, and raises annual Sharpe to {fmt(a['sharpe_without_worst'],4)}. This is a concentration diagnostic, not a reason to delete the crash.")
p('<b>(b) Predict.</b> I predicted more than half the long-only tangency weight in PUTW, and historical 1% ES substantially above the matched-normal ES. Frequent premium income looks efficient in mean-variance space, while one or two losses carry information about the insurance payoff that a normal approximation loses.')
h('(c) Maximum-Sharpe portfolios')
weights=rows('p4_weights')
tbl(['Portfolio','EQ','BOND','STK','PUTW','Ann. Sharpe'],[[short[a['portfolio']]]+[pc(a[k],2) for k in ['EQ','BOND','STK','PUTW']]+[fmt(a['sharpe'],4)] for a in weights],[120,76,76,76,80,84])
p('Unconstrained tangency uses w proportional to inverse(covariance)(mean-rf), normalized to one. The long-only and 20%-cap cases maximize the same ratio under explicit bounds using SLSQP. The cap binds at 20%; the unrestricted long-only portfolio holds 73.01% PUTW, confirming the prediction. Its tiny STK allocation should not be read as precise economic information.')
fig('p4_frontier',420)

# Page 10
page();title('4. Portfolio tails and downside beta - Fit / Reconcile')
p('<b>(c)</b> The previous chart solves minimum variance for each feasible target mean under long-only bounds. The solid upper branch is efficient; the dotted lower branch is dominated. The capital market line passes through the monthly-risk-free equivalent of 3% annually and the long-only tangency portfolio. It represents mixes with a risk-free asset, including levered extensions if borrowing is allowed; the risky-only frontier itself remains long only.')
h('(d) Monthly risk as a percentage of portfolio value')
r4=rows('p4_risk')
tbl(['Portfolio','Tail','Hist VaR','Hist ES','Normal VaR','Normal ES'],[[short[a['portfolio']],pc(a['alpha'],0)]+[pc(a[k],4) for k in ['historical_VaR','historical_ES','normal_VaR','normal_ES']] for a in r4],[120,50,85,85,86,86])
p('The matched normal uses each portfolio\'s observed monthly total-return mean and n-1 sd. Historical risk uses the realized portfolio returns, preserving the joint crash months. For 1% ES, 240 observations provide only 2.4 months of tail mass: the worst two plus 40% of the third. This is a noisy tail estimate, but the realized concentration is economically significant.')
h('(e) PUTW CAPM on monthly excess returns')
cap=rows('p4_capm')
tbl(['Sample','n','Alpha / mo','Alpha t','Beta','R-squared'],[[('All months' if a['sample']=='All_months' else 'EQ excess < -5%'),a['n'],pc(a['alpha'],4),fmt(a['alpha_t'],4),fmt(a['beta'],4),fmt(a['R2'],4)] for a in cap],[150,40,90,80,75,77])
p('The downside regression is refitted with its own intercept on the 26 months for which EQ excess return is below -5%; it is not a through-origin slope. Its intercept is a local linear extrapolation on a selected regime, not a second standalone estimate of skill. Reported t statistics use ordinary homoskedastic OLS standard errors.')
p('<b>(f)</b> The optimizer bought cash-secured short-put insurance exposure. PUTW does not borrow, but cash collateral prevents funding leverage rather than removing the put\'s nonlinear loss. Its frequent small gains and relatively low unconditional variance produce a high Sharpe. Mean and variance do not describe whether variance arises from many ordinary moves or a few severe losses. Historical long-only 1% ES is <b>9.2050%</b>, versus normal ES <b>2.8769%</b>, about 3.20 times as large. The 20% cap reduces historical 1% ES to 5.4910%, at a lower Sharpe of 0.5902.')
p('ES is coherent: monotonicity respects larger losses, cash translation shifts risk dollar for dollar, positive homogeneity scales it with position size, and subadditivity recognizes diversification. Its worst-tail averaging directly retains PUTW\'s insurance losses; it does not reward hiding a severe loss beyond the VaR cutoff. Coherence alone does not guarantee a precise empirical estimate or immunity to missing crash scenarios. Sharpe is a performance ratio, not a coherent loss-risk measure.')
p('<b>(g)</b> PUTW\'s all-month CAPM alpha of 0.2745% per month (t = 3.05) is not convincing evidence of skill. Its beta is only 0.1848 overall but 0.9428 in bad equity months, consistent with a put\'s exposure becoming stock-like when exercised. A single constant-beta regression mistakes a nonlinear payoff and its compensation for residual alpha. The richer payoff explanation is more credible than treating the intercept as free return.')

# Page 11
page();title('4. Mean estimation and weight uncertainty (h)')
p('Under an independent-month approximation, SE(mean) = sample sd/sqrt(240). Approximate 95% intervals use mean +/- 1.96 SE; dependence and changing regimes would require further adjustments. These intervals describe uncertainty in each unconditional mean, not uncertainty in the full optimized portfolio.')
tbl(['Asset','Mean / mo','SE / mo','95% mean interval','Mean excess / SE'],[[a['asset'],pc(a['mean'],4),pc(a['mean_se'],4),pc(n(a['mean'])-1.96*n(a['mean_se']),4)+' to '+pc(n(a['mean'])+1.96*n(a['mean_se']),4),fmt((n(a['mean'])-.0025)/n(a['mean_se']),3)] for a in assets],[65,95,95,170,87])
p('The standard errors are about 0.3180%, 0.0869%, 0.6020%, and 0.1068% per month for EQ, BOND, STK, and PUTW. The mean excess returns are small relative to that uncertainty, especially for the first three. Twenty years of monthly returns sounds large, but it does not make expected-return differences precise; it also supplies very few independent crash observations.')
h('Row-bootstrap sensitivity: 500 resamples, seed 5454026')
p('Resample entire monthly rows, preserving contemporaneous correlations, then re-estimate means, covariance, and weights. The following 5th-95th percentile ranges describe weight instability under this empirical resampling exercise. They are not confidence intervals for a known true optimal portfolio. The full draw table also retains unconstrained weight sensitivities.')
boot=rows('p4_bootstrap_intervals')
tbl(['Portfolio','Asset','5th percentile','Median','95th percentile'],[[short[a['portfolio']],a['asset'],pc(a['p05'],2),pc(a['median'],2),pc(a['p95'],2)] for a in boot],[130,65,105,105,107])
p('Long-only PUTW varies from about 20.4% to 98.9%, while BOND ranges from 0% to 68.3%. Even the apparently dominant PUTW allocation is far from pinned down. The unconstrained EQ weight changes sign across resamples; I would not trust the fitted -4.39% short as a stable investment conclusion. Very small positive STK weights are similarly fragile.')
p('<b>Opinion:</b> I would trust the qualitative need for diversification and an explicit insurance-exposure limit more than any exact tangency weight. The capped solution\'s PUTW weight is stable because the constraint binds, not because the data identify 20% as uniquely optimal. The bond-heavy capped allocation is a useful constrained scenario, but its exact 63.62% BOND weight remains uncertain. I would combine factor-informed mean estimates, stress scenarios, and exposure limits before implementing these sample weights.')
p('A row bootstrap cannot create an unseen crash and assumes independent months. It can omit or repeat the handful of observed disasters, which is precisely why the insurance allocation varies so much. This exercise supports the uncertainty conclusion; it does not certify the portfolio\'s future tail safety.')

# Page 12
page();title('5. Alpha or factor exposure? - Predict (a-b)')
p('Join the fund and factor data by date, requiring one-to-one matches. All 1,257 fund dates from 1996-10-01 through 2001-09-28 match. Fund returns are already decimals; all Fama-French and momentum columns, including RF, are divided by 100 exactly once. Momentum\'s trailing spaces in its column name are stripped. The dependent variable is fund total return minus same-day RF; Mkt-RF is already an excess return.')
p('I use 255 trading days per year consistently with Problem 3. Daily alphas and premia are annualized by multiplication by 255; this is arithmetic annualization, not a compounded forecast. Fund volatility below is daily excess-return sample sd. Daily total-return sd is separately used for the stress-loss normalization in (e).')
h('(a) Preliminary performance and CAPM')
fs=rows('p5_fund_stats');co=rows('p5_coefficients');mt=rows('p5_model_metrics')
def coef(fund,model,term):return next(a for a in co if a['fund']==fund and a['model']==model and a['term']==term)
def metric(fund,model):return next(a for a in mt if a['fund']==fund and a['model']==model)
tbl(['Fund','Mean excess / day','Daily excess sd','Ann. Sharpe','CAPM alpha / day','Alpha t'],[[a['fund'],pc(a['mean'],5),pc(a['sd'],4),fmt(a['sharpe'],4),pc(coef(a['fund'],'CAPM','alpha')['coefficient'],5),fmt(metric(a['fund'],'CAPM')['alpha_t'],4)] for a in fs],[60,105,95,75,110,67])
p('FUND1 initially looks more skilled: Sharpe 0.8479 versus 0.7500, and CAPM daily alpha 0.04407% with t = 3.20 versus FUND2\'s 0.02449% with t = 1.83. This is the impression to test, not the final conclusion. CAPM betas are 0.8469 and 0.4288, so higher raw return alone would not establish superior skill.')
h('(b) CAPM residual correlations and predictions')
cr=rows('p5_capm_residual_correlations')
tbl(['Fund','Residual vs SMB','Residual vs HML','Residual vs Mom'],[[f]+[fmt(next(a for a in cr if a['fund']==f and a['factor']==k)['correlation'],5) for k in ['SMB','HML','Mom']] for f in ['FUND1','FUND2']],[80,145,145,142])
p('FUND1\'s residual correlation with momentum is 0.8015, far larger than its HML correlation of 0.2052 or SMB correlation of -0.0612. I predict that adding factors, especially momentum, will sharply reduce its alpha and improve explanatory power. Its CAPM residual contains a highly systematic component.')
p('FUND2 has a 0.3295 residual correlation with SMB, but weak HML (-0.1206) and momentum (0.0530) correlations. I predict a smaller alpha change and tentatively a reduction. This direction is uncertain: residual correlation identifies omitted exposure, while the intercept change also depends on the omitted factors\' sample premia and correlations with included factors. A positive loading on a negative-premium factor can make the expanded alpha rise.')
h('Regression and information-criterion conventions')
p('All models include an intercept and use the same 1,257 dates. Standard errors are conventional OLS: residual variance SSE/(n-k_mean), multiplied by the inverse design cross-product. The alpha t statistic is alpha/SE(alpha). For AICc, Gaussian log likelihood uses the MLE variance SSE/n, and parameter count includes both the regression coefficients and the residual variance: k = 3, 5, and 6 for CAPM, FF3, and Carhart4. Lower AICc is better within a fund; cross-fund AICc is not a skill comparison.')
p('These t statistics are conditional on the model and ordinary OLS assumptions. The assignment asks for model comparisons; statistically persistent alpha would also require checking stability and inference under return dependence or heteroskedasticity. An in-sample t near two alone cannot guarantee skill.')

# Page 13
page();title('5. Factor model estimates - Fit (c)')
p('Every coefficient is shown with its standard error. Alpha is in decimal return per day; factor betas are dimensionless. The same units apply to standard errors. Model fits use excess fund returns.')
tbl(['Fund / model','Term','Coefficient','Standard error'],[[a['fund']+' / '+a['model'],a['term'],fmt(a['coefficient'],9),fmt(a['se'],9)] for a in co],[160,90,130,132])
tbl(['Fund','Model','Alpha t','R-squared','AICc'],[[a['fund'],a['model'],fmt(a['alpha_t'],5),fmt(a['R2'],6),fmt(a['AICc'],4)] for a in mt],[80,100,100,110,122])
p('FUND1 decisively favors Carhart4: R-squared rises from 0.8279 to 0.9563 and AICc falls to -11530.41. Its alpha t collapses to 0.042. FUND2 favors FF3 on AICc (-10030.94 versus -10029.13 for Carhart4); the negligible momentum beta adds little fit and does not justify its parameter penalty. The alpha remains positive in both expanded models.')

# Page 14
page();title('5. Expected returns and momentum stress - Fit (d-e)')
a=s['p5'];p(f"<b>(d)</b> The most recent ten years means the last ten <b>calendar</b> years available in the factor files: ({a['recent_start_exclusive']}, {a['recent_end']}], {a['recent_n']:,} common factor dates. It does not mean the last ten years ending at the funds' 2001 sample date or today's date. These later factor observations are used only for the requested expected-return exercise, not to estimate the fund betas.")
p('Expected annual total return without alpha = 255[mean(RF) + beta_M mean(Mkt-RF) + beta_S mean(SMB) + beta_H mean(HML) + beta_Mom mean(Mom)]. With alpha, add 255 x fitted daily four-factor alpha. Average RF is part of the total-return forecast. I use arithmetic premia rather than compounding separate factors.')
fp=rows('p5_factor_premia')
tbl(['Factor','Recent 10y daily mean','Recent 10y annual','Fund-sample annual'],[[a['factor'],pc(a['recent10_daily'],6),pc(a['recent10_annual'],4),pc(a['sample_annual'],4)] for a in fp],[80,160,135,137])
ex=rows('p5_expected_returns')
tbl(['Fund','Expected, no alpha','Expected, with alpha','Realized arithmetic','Realized geometric'],[[a['fund'],pc(a['without_alpha'],4),pc(a['with_alpha'],4),pc(next(z for z in fs if z['fund']==a['fund'])['annual_realized_arithmetic'],4),pc(next(z for z in fs if z['fund']==a['fund'])['annual_realized_geometric'],4)] for a in ex],[60,115,115,111,111])
p('Realized arithmetic is 255 times the sample mean total return; realized geometric is product(1+r)^(255/n)-1 and is labeled separately. Expected returns use four-factor betas even for FUND2, as requested in this part. The with-alpha result assumes the fitted alpha persists, which is a substantive assumption.')
h('(e) Worst momentum shock in the complete factor file')
p(f"The worst daily momentum return is <b>{pc(s['p5']['momentum_worst'],2)} on {s['p5']['momentum_worst_date']}</b>. Set all other factor returns to zero and calculate factor-only P&amp;L = $10,000,000 x beta_Mom x (-0.1844). I omit alpha, RF, and residual noise to isolate this specified one-factor shock.")
st=rows('p5_momentum_stress')
tbl(['Fund','Momentum beta','One-day loss ($)','Daily total sd','Loss / daily sd'],[[a['fund'],fmt(a['momentum_beta'],6),money(a['loss']),pc(a['daily_total_sd'],4),fmt(a['loss_in_daily_sd'],4)+' x'] for a in st],[60,115,135,100,102])
p('FUND1 loses about $725,709, or 7.2571% of capital and 6.1787 times its daily total-return sd. FUND2 loses about $10,758, or 0.1076% and 0.1497 daily sd. This large contrast is consistent with the momentum loading hidden in FUND1\'s CAPM residual. The denominator is a return standard deviation converted to dollars on the same $10 million exposure; it is not the regression residual sd.')
p('This historical factor shock is a conditional linear exposure calculation, not a claim that the funds existed in 1939 or that their betas would stay constant in such a crash. It also does not include residual or liquidity losses. It is useful precisely because the two funds\' similar-looking headline Sharpe ratios conceal very different sources of risk.')

# Page 15
page();title('5. What the alpha means - Reconcile (f-h)')
h('(f) Prediction check and FUND1 alpha accounting')
p('FUND1 agrees with my prediction: CAPM alpha falls from 0.000440660 per day to 0.000002938 in Carhart4, effectively zero relative to SE = 0.000069847. FF3 only partially removes it (alpha = 0.000296309); momentum supplies the missing major exposure. FUND2 contradicts my tentative direction: its alpha rises from 0.000244907 to 0.000272797. I had not accounted for the negative sample SMB premium: its positive SMB exposure depresses observed returns, so controlling for it raises alpha.')
p('For an OLS regression with an intercept, mean excess return = alpha + sum(beta x mean factor). Therefore alpha_CAPM - alpha_4 = (beta_M,4-beta_M,CAPM) mean(Mkt-RF) + beta_S mean(SMB) + beta_H mean(HML) + beta_Mom mean(Mom). The following daily basis-point contributions reproduce the alpha change, including the change in market beta.')
de=rows('p5_alpha_decomposition')
tbl(['Contribution to CAPM minus four-factor alpha','FUND1 (bp/day)','FUND2 (bp/day)'],[[label]+[fmt(10000*n(next(a for a in de if a['fund']==f)[key]),5) for f in ['FUND1','FUND2']] for label,key in [('Changed market beta','market_beta_change'),('SMB exposure','SMB'),('HML exposure','HML'),('Momentum exposure','Mom'),('Total alpha decrease','alpha_drop')]],[290,110,112])
p('FUND1\'s 4.37722 bp/day alpha decrease is primarily momentum (3.07703 bp) plus value (1.08057 bp), with market-beta adjustment +0.22097 bp and essentially zero SMB contribution. In this sample, annualized momentum and HML premia are 19.9375% and 9.4778%. Those are factor rewards, not fund-specific alpha. The accounting matches to floating-point precision.')
h('(g) APT and an opinion about fees')
p('Under a factor-pricing view such as APT, expected returns compensate exposure to priced systematic risks. Alpha is the expected-return residual relative to the selected factors and their premia. It is model-relative: an omitted priced exposure can appear as alpha, so a positive intercept in an underspecified regression is not automatically arbitrage or skill. A convincing residual return needs a sufficiently rich model and a reason to believe it persists after costs.')
p('<b>Opinion:</b> Between these funds I would consider paying an alpha fee for FUND2, conditional on further evidence of persistence and net-of-fee implementation, and would not pay FUND1 an alpha fee for return explained by market, value, and momentum exposure. I would use Carhart4 as the common attribution comparison because it catches FUND1\'s hidden momentum; FF3 is the parsimonious AICc winner for FUND2 and gives the same qualitative conclusion. FUND2\'s four-factor alpha t of 2.15 is suggestive, not proof, and its four-factor R-squared of 0.6149 leaves substantial residual risk.')
h('(h) Expected returns for an optimizer')
p('FUND1\'s realized arithmetic return is 20.9001% annually, well above the 13.7050% forecast without alpha from recent factor premia. Its near-zero alpha means this gap is mostly the difference in factor environments and RF, rather than persistent skill. FUND2 realizes 13.6047%; its factor-only forecast is 6.8915%, and including fitted alpha raises it to 13.8478%. That numerical closeness does not validate persistence because the alpha itself was fitted in the realized sample.')
p('A factor model can make optimizer inputs more interpretable by tying expected returns to systematic exposures and explicitly separating any assumed alpha. It cannot remove uncertainty in premia, beta stability, or manager skill. Here, RF and factor premia differ markedly between the 1996-2001 fund sample and the latest decade in the files. Feeding one sample\'s raw mean or an optimistic alpha into the Problem 4 optimizer can create concentrated, unstable weights; factor-based forecasts should be accompanied by conservative alpha assumptions and sensitivity to alternative premia.')
p('<b>Reproducibility:</b> README.md gives installation and run commands. preview.py regenerates the pre-fit diagnostics; run_analysis.py regenerates numerical CSVs and figures; tests/test_methods.py checks pricing identities, a full binary-tree reference, Greeks, tail integration, and saved accounting; build_report.py generates this PDF. The data hashes and environment versions are included. Full-precision outputs support every rounded table in this report.',True)


def footer(canvas,doc):
    canvas.saveState();canvas.setStrokeColor(colors.HexColor('#bed0dc'));canvas.line(42,35,570,35);canvas.setFont('Helvetica',8);canvas.setFillColor(blue);canvas.drawString(42,22,'FinTech 545 | Assignment 3');canvas.drawRightString(570,22,str(doc.page));canvas.restoreState()

def main():
    doc=SimpleDocTemplate(str(B/'Assignment3_Responses.pdf'),pagesize=(612,792),rightMargin=42,leftMargin=42,topMargin=38,bottomMargin=48,title='Assignment 3 - Options, Option Risk, and Portfolio Construction',author='Chuyu Qin')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    import reportlab,platform
    (B/'report_environment.json').write_text(json.dumps({'python':platform.python_version(),'reportlab':reportlab.Version},indent=2))
if __name__=='__main__':main()
