# 🏨 AtliQ Grands — Hospitality Revenue & Occupancy EDA

> **Domain:** Hospitality | **Tools:** Python · Pandas · Matplotlib · Jupyter Notebook  
> **Diploma:** Python for Data Science — Codebasics | **Status:** ✅ Completed & Validated

---

## 📋 Business Problem

AtliQ Grands, a luxury hotel chain operating across **four Indian cities**, has been losing market share and revenue due to ineffective decision-making. Management requested a data-driven analysis of booking and occupancy data to uncover patterns and support strategic recovery.

This project performs a complete **Exploratory Data Analysis (EDA)** structured across four phases — from raw data exploration through to actionable business insight generation.

---

## 🗂️ Dataset Overview

| File | Type | Records | Description |
|---|---|---|---|
| `dim_date.csv` | Dimension | 92 | Calendar with month labels, week numbers, day type |
| `dim_hotels.csv` | Dimension | 25 | Property names, categories, cities |
| `dim_rooms.csv` | Dimension | 4 | Room IDs and class (Standard, Elite, Premium, Presidential) |
| `fact_bookings.csv` | Fact | 134,590 | Individual booking records with revenue and guest ratings |
| `fact_aggregated_bookings.csv` | Fact | ~9,000 | Daily aggregated bookings and capacity by property/room |
| `new_data_august.csv` | Fact | Extension | August incremental load |

**Scope:** 3 months (May–Jul 2022) + August extension · 4 cities · 25 properties · 4 room classes

---

## 🔬 Project Phases

### Phase 1 — Data Exploration `01_data_exploration.ipynb`

Key questions answered:
- How many booking records and columns exist? → `df.shape`
- What room categories and booking platforms are in the data? → `.unique()`
- How many bookings per platform? → `.value_counts()` + horizontal bar chart
- What are the statistical ranges of revenue and ratings? → `.describe()`

**Key observations:**
- Average guest rating is low — above 4.0 is the target
- `no_guests` contains negative values — data error flagged
- `revenue_generated` max value is abnormally high — outlier investigation needed
- 4 cities, multiple hotel categories confirmed in `dim_hotels`

---

### Phase 2 — Data Cleaning `02_data_cleaning.ipynb`

| Issue | Detection Method | Resolution |
|---|---|---|
| Negative `no_guests` | `df[df.no_guests <= 0]` | Filtered out (`> 0` only) |
| Revenue outliers in `revenue_generated` | Mean + 3×Std Dev rule | Removed rows above upper limit |
| Revenue outliers in `revenue_realized` | Mean + 3×Std Dev rule | **Retained** — RT4 (Presidential) rooms validated as legitimately high-value |
| Null values in `ratings_given` | `.isnull().sum()` | **Retained** — 77,897 NaN rows represent unrated bookings, not errors |
| Null values in `capacity` | `.isna().sum()` | Filled with **median** (robust to outliers) |
| `successful_bookings > capacity` | Boolean mask | Removed — 0.07% of rows, logically invalid |

**Reasoning highlight:** RT4 Presidential rooms have a 3×Std Dev upper limit of ₹50,583 vs. actual max of ₹45,220 — no outliers confirmed after segment-level analysis.

---

### Phase 3 — Data Transformation `03_data_transformation.ipynb`

**Occupancy % calculation:**
```python
# Clean first, then calculate
df_agg_bookings = df_agg_bookings[df_agg_bookings.successful_bookings <= df_agg_bookings.capacity]
df_agg_bookings['occ_pct'] = (df_agg_bookings['successful_bookings'] / df_agg_bookings['capacity'] * 100).round(2)
```

**Dimension enrichment via merges:**
```python
df = pd.merge(df_agg_bookings, df_rooms, left_on='room_category', right_on='room_id')
df = pd.merge(df, df_hotels, on='property_id')
df = pd.merge(df, df_date, left_on='check_in_date', right_on='date')
```

**Incremental data load:**
```python
latest_df = pd.concat([df, df_august], ignore_index=True, axis=0)
```

---

### Phase 4 — Insights Generation `04_insights_generation.ipynb`

Seven business questions answered with code and visualisations.

---

## 📊 Key Business Insights

### 1️⃣ Occupancy by Room Class

```python
df.groupby('room_class')['occ_pct'].mean().round(2)
```

| Room Class | Avg Occupancy % |
|---|---|
| Presidential | ~59% |
| Premium | ~58% |
| Elite | ~58% |
| Standard | ~57% |

> Occupancy is remarkably consistent across all room classes — no tier underperforms significantly.

---

### 2️⃣ Occupancy by City

```python
df.groupby('city')['occ_pct'].mean().round(2)
```

| City | Avg Occupancy % |
|---|---|
| Delhi | ~62% 🏆 |
| Hyderabad | ~58% |
| Mumbai | ~57% |
| Bangalore | ~56% |

> Delhi leads occupancy across all months. Bangalore is the weakest market — warrants pricing or marketing strategy review.

---

### 3️⃣ Weekday vs. Weekend Occupancy

```python
df.groupby('day_type')['occ_pct'].mean().round(2)
```

| Day Type | Avg Occupancy % |
|---|---|
| Weekend | ~73% 🏆 |
| Weekday | ~51% |

> Weekend occupancy is ~22 percentage points higher. Business travel demand (weekday) is significantly underperforming — corporate package strategy is recommended.

---

### 4️⃣ June Occupancy by City

```python
df_june_22 = df[df['mmm yy'] == 'Jun 22']
df_june_22.groupby('city')['occ_pct'].mean().round(2).sort_values(ascending=False)
```

> Delhi and Hyderabad outperform in June. Ranking is consistent with overall city occupancy trend.

---

### 5️⃣ Revenue by Hotel Category

```python
df_bookings_all.groupby('category')['revenue_realized'].sum()
```

| Category | Revenue |
|---|---|
| Luxury | ₹1,052,569,562 🏆 |
| Business | ₹655,967,037 |

> Luxury properties generate **60% more revenue** than Business properties — strong justification for continued investment in the luxury segment.

---

### 6️⃣ Revenue by City

```python
df_bookings_all.groupby('city')['revenue_realized'].sum()
```

> Revenue distribution mirrors occupancy rankings — Delhi and Mumbai are top revenue contributors.

---

### 7️⃣ Monthly Revenue Trend

```python
df_bookings_all.groupby('mmm yy')['revenue_realized'].sum()
```

| Month | Revenue |
|---|---|
| May 22 | ₹581,930,666 🏆 |
| Jul 22 | ₹572,908,208 |
| Jun 22 | ₹553,932,355 |

> May is the strongest revenue month. June shows a dip — possible seasonal softening or competitive pressure worth monitoring.

---

### 8️⃣ Revenue by Booking Platform (Pie Chart)

```python
rev_platform = df_bookings_all.groupby('booking_platform')['revenue_realized'].sum().round(2)
rev_platform.plot(kind='pie', autopct='%.1f%%', startangle=90)
plt.title('Revenue Realized per Booking Platform')
```

| Platform | Revenue Share |
|---|---|
| others | ~41% ⚠️ |
| makeyourtrip | ~20% |
| logtrip | ~11% |
| Remaining platforms | 5–9% each |

> **Data quality flag:** `others` accounts for 41% of revenue — an unclassified channel that requires investigation. Among named platforms, makeyourtrip leads. The remaining channel mix is well-diversified.

---

## ⚙️ Technical Notes

### Date Conversion — Mixed Format Handling

The project required resolving **inconsistent date formats** across source files — a common real-world data quality issue when multiple systems feed into a single pipeline.

| DataFrame | Raw Format | Format String |
|---|---|---|
| `df_date` | `01-May-22` | `format='%d-%b-%y'` |
| `df_bookings_all` | `1/5/2022` and `13-05-22` (mixed) | `format='mixed', dayfirst=True` |

```python
# Explicit format — clean, consistent data (recommended for production)
df_date['date'] = pd.to_datetime(df_date['date'], format='%d-%b-%y')

# Mixed format — real-world inconsistency from concatenated sources
df_bookings_all['check_in_date'] = pd.to_datetime(
    df_bookings_all['check_in_date'], format='mixed', dayfirst=True
)
```

**Key insight:** Once converted to `datetime64`, the original string format is irrelevant. Pandas compares moments in time — not text strings — making date-based merges reliable across source inconsistencies.

### Outlier Strategy

| Column | Method | Decision |
|---|---|---|
| `revenue_generated` | Mean + 3×Std Dev | Removed outliers |
| `revenue_realized` | Mean + 3×Std Dev per segment | Retained — RT4 validated |
| `ratings_given` | Null count | Retained NaN — unrated ≠ invalid |
| `capacity` | Null count | Filled with median |

---

## 🗃️ Repository Structure

```
atliq-grands-hospitality-eda/
│
├── datasets/
│   ├── dim_date.csv
│   ├── dim_hotels.csv
│   ├── dim_rooms.csv
│   ├── fact_bookings.csv
│   ├── fact_aggregated_bookings.csv
│   └── new_data_august.csv
│
├── 01_data_exploration.ipynb
├── 02_data_cleaning.ipynb
├── 03_data_transformation.ipynb
├── 04_insights_generation.ipynb
├── hotels_analysis.ipynb
└── README.md
```

---

## 🚀 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/jarrano/atliq-grands-hospitality-eda.git
cd atliq-grands-hospitality-eda

# 2. Install dependencies
pip install pandas matplotlib jupyter

# 3. Launch Jupyter
jupyter notebook

# 4. Run notebooks in order: 01 → 02 → 03 → 04
```

---

## 👤 Author

**Jorge Arraño (艾宥承)**  
Senior Digital Transformation Lead | AI & Analytics  
📍 Kaohsiung, Taiwan  
🔗 [LinkedIn](https://www.linkedin.com/in/jarrano) · [GitHub](https://github.com/jarrano) · [Codebasics Portfolio](https://codebasics.io/portfolio/Jorge-Arrao)
