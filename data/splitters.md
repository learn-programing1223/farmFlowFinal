# Data Splitting Requirements

## CRITICAL REQUIREMENT: Prevent Data Leakage
**IBM Research Finding**: Data leakage is the #1 cause of overoptimistic accuracy in agricultural ML

## MANDATORY CONSTRAINTS

### Images from Same Source CANNOT Appear in Different Splits
- Images from same plant → same split only
- Images from same location → same split only  
- Images from same time period → same split only
- Images from same plant group → same split only

### Required Grouping Strategy
```
Group Key = plant_id + location + date
```
- **plant_id**: Unique identifier for each plant
- **location**: GPS coordinates or field section
- **date**: Capture date (group by week/month)

### Split Ratios
- **Training**: 70%
- **Validation**: 15%
- **Test**: 15%

## VALIDATION CHECKS

### Leakage Detection
1. **No duplicate images** across splits (check MD5 hash)
2. **No shared plant_ids** across splits
3. **No shared location+date** combinations across splits
4. **Alert if**: validation accuracy > training accuracy + 5%

### Statistical Validation
- Maintain class balance (±2%) across all splits
- Ensure each split has all disease categories
- Verify similar lighting condition distribution
- Check disease stage distribution

## IMPLEMENTATION REQUIREMENTS

### Data Structure
Each image must have metadata:
- `image_path`: File location
- `disease_label`: Ground truth class
- `plant_id`: Unique plant identifier
- `location`: GPS or field section
- `date`: Capture timestamp
- `lighting_condition`: sun/shade/cloudy
- `disease_stage`: early/mid/late

### Stratification Strategy
- Primary: Disease class
- Secondary: Lighting conditions
- Tertiary: Disease stage

### Edge Cases
- Unknown disease samples: Distribute proportionally
- Single-image plants: Assign to training (document count)
- Missing metadata: Quarantine for manual review

## PERFORMANCE IMPACT
Proper splitting prevents:
- 15-20% accuracy overestimation
- False confidence in field performance
- Model failure on new plants/locations

## VALIDATION METRICS
- Cross-split similarity score (should be <0.1)
- Metadata distribution tests
- Time-based validation (older→train, newer→test)

## Related Research
- IBM paper on agricultural ML pitfalls
- Data Pipeline Architecture section
- Stanford paper on data leakage in medical imaging

## Implementation Freedom
Let Claude Code determine optimal implementation within these strict constraints