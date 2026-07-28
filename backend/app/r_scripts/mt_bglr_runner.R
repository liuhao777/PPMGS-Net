args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 7) {
  stop("Usage: mt_bglr_runner.R <x_csv> <y_csv> <output_dir> <nIter> <burnIn> <thin> <seed>")
}

x_path <- args[[1]]
y_path <- args[[2]]
output_dir <- args[[3]]
n_iter <- as.integer(args[[4]])
burn_in <- as.integer(args[[5]])
thin <- as.integer(args[[6]])
seed <- as.integer(args[[7]])

dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
set.seed(seed)

if (!requireNamespace("BGLR", quietly = TRUE)) {
  stop("The optional MT-BGLR baseline requires the R package 'BGLR'. Install it before running this baseline.")
}
suppressPackageStartupMessages(library(BGLR))

X_df <- read.csv(x_path, check.names = FALSE)
Y_df <- read.csv(y_path, check.names = FALSE, na.strings = c("", "NA", "NaN", "nan", "null", "None"))
X <- as.matrix(X_df)
Y <- as.matrix(Y_df)
storage.mode(X) <- "double"
storage.mode(Y) <- "double"

if (ncol(Y) < 2) {
  stop("BGLR::Multitrait requires at least two traits.")
}
if (nrow(X) != nrow(Y)) {
  stop("X and Y must have the same number of rows.")
}

marker_count <- max(ncol(X), 1)
K <- tcrossprod(X) / marker_count
K <- (K + t(K)) / 2
diag(K) <- diag(K) + 1e-6

fit_multitrait <- function(cov_type) {
  save_prefix <- file.path(output_dir, paste0("bglr_", tolower(cov_type), "_"))
  ETA <- list(
    genomic = list(
      K = K,
      model = "RKHS",
      Cov = list(type = cov_type)
    )
  )
  BGLR::Multitrait(
    y = Y,
    ETA = ETA,
    resCov = list(type = cov_type),
    nIter = n_iter,
    burnIn = burn_in,
    thin = thin,
    verbose = FALSE,
    saveAt = save_prefix
  )
}

used_cov_type <- "UN"
fit <- tryCatch(
  fit_multitrait("UN"),
  error = function(e) {
    message("MT-BGLR UN covariance failed: ", conditionMessage(e))
    message("Retrying MT-BGLR with DIAG covariance.")
    used_cov_type <<- "DIAG"
    fit_multitrait("DIAG")
  }
)

pred <- fit$ETAHat
if (is.null(pred) && !is.null(fit$ETA[[1]]$u)) {
  mu <- fit$mu
  if (is.null(mu)) {
    mu <- rep(0, ncol(Y))
  }
  pred <- sweep(fit$ETA[[1]]$u, 2, mu, "+")
}
if (is.null(pred)) {
  stop("Could not extract BGLR predictions from fit$ETAHat or fit$ETA[[1]]$u.")
}

colnames(pred) <- colnames(Y)
write.csv(as.data.frame(pred), file.path(output_dir, "predictions_scaled.csv"), row.names = FALSE, na = "")

if (!is.null(fit$mu)) {
  mu_out <- as.data.frame(matrix(fit$mu, nrow = 1))
  colnames(mu_out) <- colnames(Y)
  write.csv(mu_out, file.path(output_dir, "mu_scaled.csv"), row.names = FALSE, na = "")
}
if (!is.null(fit$resCov$R)) {
  res_cov <- as.data.frame(fit$resCov$R)
  colnames(res_cov) <- colnames(Y)
  rownames(res_cov) <- colnames(Y)
  write.csv(res_cov, file.path(output_dir, "residual_covariance.csv"), row.names = TRUE, na = "")
}
if (!is.null(fit$ETA[[1]]$Cov$Omega)) {
  gen_cov <- as.data.frame(fit$ETA[[1]]$Cov$Omega)
  colnames(gen_cov) <- colnames(Y)
  rownames(gen_cov) <- colnames(Y)
  write.csv(gen_cov, file.path(output_dir, "genomic_covariance.csv"), row.names = TRUE, na = "")
}

summary <- data.frame(
  key = c(
    "method",
    "r_package",
    "linear_predictor",
    "kernel",
    "trait_covariance",
    "residual_covariance",
    "samples",
    "markers",
    "traits",
    "nIter",
    "burnIn",
    "thin"
  ),
  value = c(
    "BGLR::Multitrait",
    "BGLR",
    "RKHS",
    "standardized_marker_grm_xxT_over_marker_count",
    used_cov_type,
    used_cov_type,
    as.character(nrow(X)),
    as.character(ncol(X)),
    as.character(ncol(Y)),
    as.character(n_iter),
    as.character(burn_in),
    as.character(thin)
  )
)
write.csv(summary, file.path(output_dir, "summary.csv"), row.names = FALSE, na = "")
