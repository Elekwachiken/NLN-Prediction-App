import pandas as pd
import base64
import io

def read_uploaded_file(contents, filename):
    """Reads uploaded file contents and returns a pandas DataFrame."""
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    try:
        if filename.endswith('.csv'):
            # Try reading CSV
            s = decoded.decode('utf-8', errors='replace')
            df = pd.read_csv(io.StringIO(s))
        elif filename.endswith(('.xls', '.xlsx')):
            xls = pd.ExcelFile(io.BytesIO(decoded))
            # Combine all sheets, add sheet name for traceability
            dfs = []
            for sheet in xls.sheet_names:
                df = xls.parse(sheet)
                df['sheet_name'] = sheet
                dfs.append(df)
            return pd.concat(dfs, ignore_index=True)
        else:
            raise ValueError("Unsupported file type. Please upload a CSV or Excel (.xlsx) file.")
    except Exception as e:
        raise ValueError(f"Failed to read file {filename}: {e}")

    # Strip column names of whitespace and lowercase for consistency
    df.columns = [col.strip().lower() for col in df.columns]

    return df



# import pandas as pd
# import base64
# import io
# import openpyxl # You'll need to install this: pip install openpyxl

# def read_uploaded_file(contents, filename):
#     """Reads uploaded file contents and returns a pandas DataFrame."""
#     content_type, content_string = contents.split(',')
#     decoded = base64.b64decode(content_string)

#     try:
#         if filename.endswith('.csv'):
#             s = decoded.decode('utf-8', errors='replace')
#             df = pd.read_csv(io.StringIO(s))
#         elif filename.endswith(('.xls', '.xlsx')):
#             # Use openpyxl for .xlsx and xlrd for .xls explicitly for large files
#             if filename.endswith('.xlsx'):
#                 # Read XLSX using openpyxl's read-only mode for memory efficiency
#                 workbook = openpyxl.load_workbook(io.BytesIO(decoded), read_only=True)
#                 all_dfs = []
#                 for sheet_name in workbook.sheetnames:
#                     sheet = workbook[sheet_name]
#                     # Read rows from the sheet
#                     data = []
#                     header = [cell.value for cell in sheet[1]] # Assuming header in first row
#                     for row_idx, row in enumerate(sheet.iter_rows(min_row=2), 2): # Start from second row for data
#                         row_data = [cell.value for cell in row]
#                         data.append(row_data)
                    
#                     if data: # Only create DataFrame if there's data
#                         df_sheet = pd.DataFrame(data, columns=header)
#                         df_sheet['sheet_name'] = sheet_name # Add sheet name
#                         all_dfs.append(df_sheet)
                
#                 if all_dfs:
#                     df = pd.concat(all_dfs, ignore_index=True)
#                 else:
#                     df = pd.DataFrame() # Empty DataFrame if no data found
#             elif filename.endswith('.xls'):
#                 # For old .xls files, pandas default (xlrd) is usually fine, but if memory is issue
#                 # you might need to look into other libraries or convert them to xlsx.
#                 # For now, stick with pandas for .xls, as xlrd doesn't have direct streaming like openpyxl.
#                 df = pd.read_excel(io.BytesIO(decoded), engine='xlrd') 
#                 # This will still load the whole sheet, so .xls might still be problematic if huge.
#                 # Consider converting to .xlsx first if user insists on large .xls.
#             else:
#                 raise ValueError("Unsupported Excel file extension.") # Should be caught by outer if

#         else:
#             raise ValueError("Unsupported file type. Please upload a CSV or Excel (.xlsx) file.")
#     except Exception as e:
#         raise ValueError(f"Failed to read file {filename}: {e}")

#     # Strip column names of whitespace and lowercase for consistency
#     if not df.empty: # Only normalize if DataFrame is not empty
#         df.columns = [col.strip().lower() for col in df.columns]

#     return df

