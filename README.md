# PRODIGY_DS_02 – Titanic Data Cleaning & EDA

**Prodigy InfoTech – Data Science Internship | Task 02**

## Objective
Clean the Titanic dataset and explore relationships between variables to find patterns and trends.

## Cleaning steps
| Issue | Fix |
|---|---|
| `Age` – 177 missing | Filled with the median age of each passenger *title* (Mr, Mrs, Miss, Master...) |
| `Embarked` – 2 missing | Filled with the mode (S) |
| `Cabin` – 687 missing (77%) | Reduced to a `Deck` letter, unknown = `U`; original column dropped |
| Duplicates | None found |
| New features | `Title`, `FamilySize`, `IsAlone`, `AgeGroup` |

Cleaned file: `titanic_cleaned.csv`

## Key findings
- Overall survival rate: **38.4%**
- **Sex** is the strongest factor: women 74.2% vs men 18.9%
- **Class**: 1st 63.0%, 2nd 47.3%, 3rd 24.2%
- **Age**: children survive most (57.5%); seniors least (22.7%)
- **Embarked**: Cherbourg 55.4%, Queenstown 39.0%, Southampton 33.9%
- Travelling **alone** lowers survival (30.4%) vs with family (50.6%)

## Run
```bash
pip install -r requirements.txt
python PRODIGY_DS_02.py
```

## Output
6 figures (`01_...png` to `06_...png`) and `titanic_cleaned.csv`
# PRODIGY_DS_02
