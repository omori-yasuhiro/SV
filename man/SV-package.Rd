\name{Stocastic Volatility Package}
\alias{SV-package}
\alias{SV}
\docType{package}
\title{
Mixture sampler for stochastic volatility models
}
\description{
\packageTitle{SV}
Mixture sampler for stochastic volatility models
}
\description{
\packageDescription{SV}
It will implement Markov chain Monte Carlo simulation for stochastic volatility models using the mixture sampler by Omori, Chib, Shephard and Nakajima (2007).
}
\details{

The DESCRIPTION file:
\packageDESCRIPTION{SV}
\packageIndices{SV}
~~ An overview of how to use the package, including ~~
~~ the most important functions ~~
}
\author{
Ryuji Hashimoto and Yasuhiro Omori. 
Maintainer: Yasuhiro Omori<omori@e.u-tokyo.ac.jp>
}
\references{
Omori, Y., Chib, S., Shephard, N., & Nakajima, J. (2007). Stochastic volatility with leverage: Fast and efficient likelihood inference. Journal of Econometrics, 140(2), 425-449.
}
\keyword{ Markov chain Monte Carlo, Mixture Sampler, Stochastic Volatility }
\seealso{
}
\examples{
svmcmc(data, TRUE)
}
