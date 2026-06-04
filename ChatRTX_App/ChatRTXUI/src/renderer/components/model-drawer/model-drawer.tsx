/* SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

import React, { useEffect, useState } from 'react'
import type { ClientAPI } from '../../../electron/preload'
import { useTranslation } from 'react-i18next'
import CustomDrawer from '../custom-drawer/custom-drawer'
import ModelCard from '../model-card/model-card'
import { HfModelBackend, ModelDetails, ModelId } from '../../../electron/types'
import {
    Backdrop,
    Box,
    Button,
    CircularProgress,
    MenuItem,
    Snackbar,
    Stack,
    TextField,
    Typography,
} from '@mui/material'
import { themeSettings } from '../../theme/theme'
import CustomLoadingBackdrop from '../custom-loading-backdrop/custom-loading-backdrop'

const clientAPI = (window as any).clientAPI as typeof ClientAPI

const HF_MODEL_TYPE_LABELS: Record<HfModelBackend, string> = {
    gguf: 'GGUF',
    onnxrt: 'ONNX Runtime GenAI',
    pytorch_rocm: 'PyTorch ROCm',
}

export default function ModelDrawer({
    openDrawer,
    onDrawerClosed,
}: {
    openDrawer: boolean
    onDrawerClosed: () => void
}) {
    const [supportedModels, setSupportedModels] = useState<ModelDetails[]>(
        clientAPI.getSupportedModels()
    )
    const [activeModel, setActiveModel] = useState<ModelId>(
        clientAPI.getActiveModel()
    )
    const [activeModelName, setActiveModeName] = useState<string>(
        clientAPI.getActiveModelDetails().name
    )

    const [modelSelectionInProgress, setModelSelectionInProgress] =
        useState<ModelId>(null)

    const [modelDownloadInProgress, setModelDownloadInProgress] =
        useState<ModelId>(null)

    const [modelInstallUnderProgress, setModelInstallUnderProgress] =
        useState<string>(null)

    const [modelDeleteUnderProgress, setModelDeleteUnderProgress] =
        useState<string>(null)
    const [toastMessage, setToastMessage] = useState<string>(null)

    const [hfRepoId, setHfRepoId] = useState<string>('')
    const [hfCompatibleModelTypes, setHfCompatibleModelTypes] = useState<
        HfModelBackend[]
    >(clientAPI.getHfCompatibleModelTypes())
    const [hfModelType, setHfModelType] = useState<HfModelBackend | null>(
        clientAPI.getHfCompatibleModelTypes()[0] ?? null
    )
    const [hfModelAddInProgress, setHfModelAddInProgress] =
        useState<boolean>(false)

    const { t } = useTranslation()

    const refreshHfCompatibleModelTypes = () => {
        const types = clientAPI.getHfCompatibleModelTypes()
        setHfCompatibleModelTypes(types)
        setHfModelType((currentType) =>
            currentType && types.includes(currentType)
                ? currentType
                : types[0] ?? null
        )
    }

    useEffect(() => {
        const supportedModelListener = clientAPI.onSupportedModelsUpdated(
            () => {
                setSupportedModels(clientAPI.getSupportedModels())
                refreshHfCompatibleModelTypes()
            }
        )

        const activeModelUpdateListener = clientAPI.onActiveModelUpdate(
            (id) => {
                setModelSelectionInProgress(null)
                setActiveModel(id)
                setActiveModeName(clientAPI.getActiveModelDetails().name)
                clientAPI.resetChat()
            }
        )

        const activeModelUpdateErrorListener =
            clientAPI.onActiveModelUpdateError((modelId) => {
                setModelSelectionInProgress(null)
            })

        const datasetchangeListener = clientAPI.onDatasetInfoUpdate(() => {
            setSupportedModels(clientAPI.getSupportedModels())
            refreshHfCompatibleModelTypes()
            clientAPI.resetChat()
        })

        const modelDownloadListener = clientAPI.onModelDownloaded((modelId) => {
            setModelDownloadInProgress(null)
            setToastMessage(
                t('modelDownloadSuccess', {
                    modelName: clientAPI.getModelDetails(modelId).name,
                })
            )
        })

        const modelDownloadErrorListener = clientAPI.onModelDownloadError(
            (modelId) => {
                setModelDownloadInProgress(null)
                setToastMessage(
                    t('modelDownloadError', {
                        modelName: clientAPI.getModelDetails(modelId).name,
                    })
                )
            }
        )

        const modelInstallListener = clientAPI.onModelInstalled((modelId) => {
            setModelInstallUnderProgress(null)
            setToastMessage(
                t('modelInstallSuccess', {
                    modelName: clientAPI.getModelDetails(modelId).name,
                })
            )
            clientAPI.resetChat()
        })

        const modelInstallErrorListener = clientAPI.onModelInstallError(
            (modelId) => {
                setModelInstallUnderProgress(null)
                setToastMessage(
                    t('modelInstallError', {
                        modelName: clientAPI.getModelDetails(modelId).name,
                    })
                )
            }
        )

        const modelDeleteListener = clientAPI.onModelDeleted((modelId) => {
            setModelDeleteUnderProgress(null)
            setToastMessage(
                t('modelUninstallSuccess', {
                    modelName: clientAPI.getModelDetails(modelId).name,
                })
            )
        })

        const modelDeleteErrorListener = clientAPI.onModelDeleteError(
            (modelId) => {
                setModelDeleteUnderProgress(null)
                setToastMessage(
                    t('modelUninstallError', {
                        modelName: clientAPI.getModelDetails(modelId).name,
                    })
                )
            }
        )

        const hfModelAddedListener = clientAPI.onHfModelAdded((repoId) => {
            setHfModelAddInProgress(false)
            setHfRepoId('')
            setToastMessage(
                t('hfModelAddSuccess', {
                    defaultValue: `Model ${repoId} added successfully`,
                    repoId,
                })
            )
        })

        const hfModelAddErrorListener = clientAPI.onHfModelAddError(
            (repoId) => {
                setHfModelAddInProgress(false)
                setToastMessage(
                    t('hfModelAddError', {
                        defaultValue: `Failed to add model ${repoId}`,
                        repoId,
                    })
                )
            }
        )

        return () => {
            supportedModelListener()
            datasetchangeListener()
            modelDownloadListener()
            modelDownloadErrorListener()
            activeModelUpdateListener()
            activeModelUpdateErrorListener()
            modelInstallListener()
            modelInstallErrorListener()
            modelDeleteListener()
            modelDeleteErrorListener()
            hfModelAddedListener()
            hfModelAddErrorListener()
        }
    }, [])

    const onModelSelectionChange = (modelId: ModelId) => {
        setModelSelectionInProgress(modelId)
        clientAPI.setActiveModel(modelId)
        setActiveModeName(clientAPI.getActiveModelDetails().name)
    }

    const onModelDownloadClick = (modelId: ModelId) => {
        clientAPI.downloadModel(modelId)
        setModelDownloadInProgress(modelId)
    }

    const onModelInstallClick = (modelId: ModelId, modelName: string) => {
        clientAPI.installModel(modelId)
        setModelInstallUnderProgress(modelName)
    }

    const onModelDeleteClick = (
        modelId: ModelId,
        modelName: string,
        backend: string
    ) => {
        if (backend === 'nims') {
            setModelDeleteUnderProgress(modelName)
        }
        clientAPI.deleteModel(modelId)
    }

    const handleToastClose = (_: any, reason: string) => {
        if (reason === 'clickaway') {
            return
        }

        setToastMessage(null)
    }

    const onAddHfModelClick = () => {
        const trimmed = hfRepoId.trim()
        const parts = trimmed.split('/')
        if (
            !trimmed ||
            parts.length !== 2 ||
            !parts[0] ||
            !parts[1]
        ) {
            setToastMessage(
                t('hfModelInvalidId', {
                    defaultValue:
                        'Please enter a valid Hugging Face model ID (e.g., meta-llama/Llama-2-7b)',
                })
            )
            return
        }
        if (!hfModelType) {
            setToastMessage(
                t('hfModelTypeUnavailable', {
                    defaultValue:
                        'No compatible Hugging Face model types are available on this hardware.',
                })
            )
            return
        }
        setHfModelAddInProgress(true)
        clientAPI.addHfModel(trimmed, hfModelType)
    }

    return (
        <CustomDrawer
            drawerHeader={t('selectAIModel')}
            openDrawer={openDrawer}
            onDrawerClosed={onDrawerClosed}
        >
            <Box sx={{ padding: '16px', marginBottom: '8px' }}>
                <Typography
                    variant="body2"
                    sx={{ marginBottom: '8px', color: 'rgba(255, 255, 255, 0.7)' }}
                >
                    {t('addHfModelLabel', {
                        defaultValue: 'Add a model from Hugging Face',
                    })}
                </Typography>
                <Typography
                    variant="body2"
                    sx={{ marginBottom: '8px', color: 'rgba(255, 255, 255, 0.5)' }}
                >
                    {t('hfModelTypeLabel', {
                        defaultValue: 'Compatible model types',
                    })}
                </Typography>
                <Stack direction="row" spacing={1} alignItems="center">
                    <TextField
                        select
                        size="small"
                        value={hfModelType ?? ''}
                        onChange={(e) =>
                            setHfModelType(e.target.value as HfModelBackend)
                        }
                        disabled={
                            hfModelAddInProgress ||
                            hfCompatibleModelTypes.length === 0
                        }
                        sx={{
                            minWidth: '220px',
                            '& .MuiInputBase-input': {
                                color: 'rgba(255, 255, 255, 0.9)',
                                fontSize: '14px',
                            },
                            '& .MuiSvgIcon-root': {
                                color: 'rgba(255, 255, 255, 0.7)',
                            },
                            '& .MuiOutlinedInput-root': {
                                '& fieldset': {
                                    borderColor: 'rgba(255, 255, 255, 0.3)',
                                },
                                '&:hover fieldset': {
                                    borderColor: 'rgba(255, 255, 255, 0.5)',
                                },
                                '&.Mui-focused fieldset': {
                                    borderColor: themeSettings.colors.brand,
                                },
                            },
                        }}
                    >
                        {hfCompatibleModelTypes.map((modelType) => (
                            <MenuItem key={modelType} value={modelType}>
                                {HF_MODEL_TYPE_LABELS[modelType]}
                            </MenuItem>
                        ))}
                    </TextField>
                    <TextField
                        size="small"
                        placeholder="owner/model-name"
                        value={hfRepoId}
                        onChange={(e) => setHfRepoId(e.target.value)}
                        disabled={
                            hfModelAddInProgress ||
                            hfCompatibleModelTypes.length === 0
                        }
                        sx={{
                            flex: 1,
                            '& .MuiInputBase-input': {
                                color: 'rgba(255, 255, 255, 0.9)',
                                fontSize: '14px',
                            },
                            '& .MuiOutlinedInput-root': {
                                '& fieldset': {
                                    borderColor: 'rgba(255, 255, 255, 0.3)',
                                },
                                '&:hover fieldset': {
                                    borderColor: 'rgba(255, 255, 255, 0.5)',
                                },
                                '&.Mui-focused fieldset': {
                                    borderColor: themeSettings.colors.brand,
                                },
                            },
                        }}
                        onKeyDown={(e) => {
                            if (e.key === 'Enter') {
                                onAddHfModelClick()
                            }
                        }}
                    />
                    <Button
                        variant="contained"
                        size="small"
                        onClick={onAddHfModelClick}
                        disabled={
                            hfModelAddInProgress ||
                            !hfRepoId.trim() ||
                            !hfModelType
                        }
                        sx={{
                            backgroundColor: themeSettings.colors.brand,
                            textTransform: 'none',
                            '&:hover': {
                                backgroundColor: themeSettings.colors.brandLight,
                            },
                        }}
                    >
                        {hfModelAddInProgress ? (
                            <CircularProgress size={20} color="inherit" />
                        ) : (
                            t('addModel', { defaultValue: 'Add' })
                        )}
                    </Button>
                </Stack>
            </Box>
            {supportedModels.map((modelDetails: ModelDetails) => (
                <div key={modelDetails.id}>
                    <ModelCard
                        modelDetails={modelDetails}
                        isActive={activeModel === modelDetails.id}
                        isModelSelectionInProgress={
                            modelSelectionInProgress === modelDetails.id
                        }
                        isModelDownloadInProgress={
                            modelDownloadInProgress === modelDetails.id
                        }
                        disableModelAction={!!modelDownloadInProgress}
                        onModelSelectionChange={onModelSelectionChange}
                        onModelDownloadClick={onModelDownloadClick}
                        onModelInstallClick={() =>
                            onModelInstallClick(
                                modelDetails.id,
                                modelDetails.name
                            )
                        }
                        onModelDeleteClick={() =>
                            onModelDeleteClick(
                                modelDetails.id,
                                modelDetails.name,
                                modelDetails.backend
                            )
                        }
                        activeModelName={activeModelName}
                    ></ModelCard>
                </div>
            ))}
            <Backdrop
                sx={{ backgroundColor: 'rgba(17, 17, 17)' }}
                open={!!modelInstallUnderProgress || !!modelDeleteUnderProgress}
            >
                <Stack
                    direction={'row'}
                    alignItems={'center'}
                    justifyContent={'center'}
                >
                    <Stack
                        direction={'column'}
                        alignItems={'center'}
                        justifyContent={'center'}
                    >
                        <CircularProgress size={48}></CircularProgress>
                        <Typography
                            sx={{
                                textAlign: 'center',
                                marginTop: '24px',
                                color: 'rgba(255, 255, 255, 0.9)',
                            }}
                            variant="body2"
                        >
                            {modelInstallUnderProgress
                                ? t('installModelProgress', {
                                      modelName: modelInstallUnderProgress,
                                  })
                                : t('uninstallingModelProgress', {
                                      modelName: modelDeleteUnderProgress,
                                  })}
                        </Typography>
                        <Typography variant="body2">
                            {t('waitFewMin')}
                        </Typography>
                    </Stack>
                </Stack>
            </Backdrop>
            <CustomLoadingBackdrop
                open={!!modelSelectionInProgress}
                loadingText={t('loadingModel')}
            ></CustomLoadingBackdrop>
            <Snackbar
                ContentProps={{
                    sx: {
                        backgroundColor: themeSettings.colors.n600,
                    },
                }}
                open={!!toastMessage}
                autoHideDuration={5000}
                onClose={handleToastClose}
                message={
                    <Typography variant="body2">{toastMessage}</Typography>
                }
            ></Snackbar>
        </CustomDrawer>
    )
}
