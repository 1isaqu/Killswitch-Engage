# How to Contribute to Killswitch Engage

First of all, thank you for your interest in contributing! The **Killswitch Engage** project is a complete recommendation system focused on the intersection of Software Engineering best practices and Machine Learning in production.

Below are the guidelines to ensure your contribution aligns with the project's standards.

---

## 1. Basic Contribution Process

1. **Fork** the repository.
2. Create a **Branch** for your feature or fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```
   or
   ```bash
   git checkout -b fix/your-fix-name
   ```
3. Make clear and descriptive **Commits**. Follow the **Conventional Commits** standard (see below).
4. Push the changes to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
5. Open a detailed **Pull Request (PR)** to the main branch (`main`).

---

## 2. Code Standards and Architecture

This project follows strict Clean Code, SOLID, and security rules. Before submitting code, ensure that:

*   **Single Responsibility**: Functions and classes should do only one thing. Avoid "God Objects".
*   **Strong Typing**: The use of **Type Hints** (`-> int`, `List[str]`, etc.) is mandatory in all function and method signatures.
*   **Docstrings**: Use the Google standard to document the "WHY" of complex functions, not the "WHAT". The code should be mostly self-explanatory.
*   **No Magic**: Avoid magic numbers or scattered literals. Use constants or configurations from `.env` / `settings.py`.
*   **Error Handling**: Throw descriptive exceptions instead of returning silently or logging `null`.

---

## 3. Commit Standard

We use **Conventional Commits** to standardize the Git history:

*   `feat:` New feature
*   `fix:` Bug fix
*   `docs:` Documentation-only changes (e.g., README, this file)
*   `style:` Code formatting or styling changes (spaces, commas, etc.)
*   `refactor:` Code refactoring that neither fixes a bug nor adds a feature
*   `perf:` Code changes focused on performance improvement
*   `test:` Adding or correcting tests
*   `chore:` Build updates, dependencies, or secondary tools

**Example:** `feat: added cGAN model for meta recommendation`

---

## 4. Security and Sensitive Data

**CRITICAL:** Never version files with credentials or keys.
*   Use the `.env` file for sensitive data.
*   There is a `.env.example` file that you should use as a template. Leave dummy values in it.
*   Do not mask database connection errors. Instead, fail-fast when trying to create the connection `pool` in `database.py`.

---

## 5. How to Run and Test Logic Locally

The project depends on external services (PostgreSQL and Redis). Before running tests or the API, you must spin up the complete environment via Docker:

```bash
# Start the database and cache containers in the background
docker-compose up -d
```

After spinning up the services, to test your contribution before making a Pull Request, follow the commands below using `pytest`:

```bash
# Run the complete test suite
pytest tests/

# Run with detailed logs
pytest -s -v tests/
```

The project uses `black` and `flake8` via `pre-commit`. Make sure to run the command below before opening the PR to ensure all formatting is correct:

```bash
pre-commit run --all-files
```

---

Thank you for your collaboration in making **Killswitch Engage** a better project!
