# Security Audit Report — Killswitch Engage

**Date:** 2026-03-06
**Status:** ✅ **PASSED WITH MINOR RECOMMENDATIONS**

---

## 1. Credentials & Secrets
- **Observation:** All sensitive variables (`DB_PASSWORD`, `SUPABASE_KEY`, `REDIS_URL`) are stored in a `.env` file and accessed via `os.getenv`.
- **Finding:** No hardcoded credentials were found in `.py`, `.ipynb`, or `.md` files.
- **Checklist Compliance:** 100%

## 2. Database Security
- **Observation:** SQL queries in `backend/app/models/queries.py` and `src/backend/routes/usuarios.py` use positional parameters (`$1`, `$2`) provided to `asyncpg`.
- **SSL:** `DB_SSL_MODE` is implemented in `config.py` with `require` as default.
- **Checklist Compliance:** 100%

## 3. API & Backend
- **CORS:** Restricted to `ALLOWED_ORIGINS` in `config.py`. Never use `*` in production.
- **Rate Limiting:** Enforced on all routes using `slowapi` (`@limiter.limit("100/minute")`).
- **Security Headers:** `SecurityHeadersMiddleware` implements HSTS, CSP (`default-src 'self'`), X-Frame-Options (`DENY`), and X-Content-Type-Options (`nosniff`).
- **Checklist Compliance:** 100%

## 4. Models & Files
- **Insecure Loading:** The project uses `joblib` and `torch.load` for model artefacts. Scripts include explicit security warnings (§ 4) about loading only trusted files.
- **Safe Parsing:** Categorisation parsing uses `ast.literal_eval` instead of `eval`.
- **Checklist Compliance:** 100% (with documented risks)

## 5. Dependencies
- **Observation:** `backend/requirements.txt` pins core dependencies (`fastapi`, `uvicorn`, etc.), but some ML libraries (`scikit-learn`, `joblib`, `lightfm`) are unpinned.
- **Recommendation:** Pin all versions in `requirements.txt` to ensure production stability and security.
- **Checklist Compliance:** 80% (Minor improvement needed)

## 6. Logs & PII
- **Observation:** `src/utils/logger.py` implements a `_SanitizingFilter` that drops log entries containing sensitive keywords with assignment operators.
- **PII:** User routes explicitly mask `usuario_id` in logs (e.g., `Session recorded for user [...]`).
- **Checklist Compliance:** 100%

---

## 📝 Recommendations
1. **Dependency Pinning:** Run `pip freeze > requirements.txt` and fix every version.
2. **cGAN Integration:** The cGAN meta-learner is implemented in `scripts/meta_learning` but not yet integrated into the `RecomendadorService` as specified in the README. Update the service to utilize the Layer 4 model.
