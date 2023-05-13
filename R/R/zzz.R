
.onLoad <- function(libname, lkgname) {
    suppressWarnings(Rcpp::loadRcppModules())
}
